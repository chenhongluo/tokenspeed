#!/usr/bin/env bash
# Lite 8P8D bounded eager serving on one 16-NPU node.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/worker_cleanup.sh"

CHECK_ONLY=0
if [[ ${1:-} == "--check" ]]; then
  CHECK_ONLY=1
  shift
fi
if (($#)); then
  echo "usage: $0 [--check]" >&2
  exit 2
fi

MODEL=${MODEL:-}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-lite}
PYTHON=${PYTHON:-python3}
PREFILL_NPUS=${PREFILL_NPUS:-0,1,2,3,4,5,6,7}
DECODE_NPUS=${DECODE_NPUS:-8,9,10,11,12,13,14,15}
WORLD_SIZE=${WORLD_SIZE:-8}
PREFILL_PORT=${PREFILL_PORT:-28346}
PREFILL_BOOTSTRAP_PORT=${PREFILL_BOOTSTRAP_PORT:-29998}
DECODE_PORT=${DECODE_PORT:-30347}
PREFILL_DIST_PORT=${PREFILL_DIST_PORT:-29579}
DECODE_DIST_PORT=${DECODE_DIST_PORT:-31580}
PREFILL_HCCL_BASE_PORT=${PREFILL_HCCL_BASE_PORT:-8282}
DECODE_HCCL_BASE_PORT=${DECODE_HCCL_BASE_PORT:-8382}
LB_HOST=${LB_HOST:-0.0.0.0}
LB_PORT=${LB_PORT:-28345}
PROMETHEUS_PORT=${PROMETHEUS_PORT:-28422}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-1024}
PREFIX_GRANULARITY=${PREFIX_GRANULARITY:-64}
LOG_DIR=${LITE_PD_LOG_DIR:-.ci-artifacts/lite-npu-pd-8p8d}

fail() {
  echo "[lite-npu-pd] $*" >&2
  exit 2
}

positive_int() {
  [[ $2 =~ ^[1-9][0-9]*$ ]] || fail "$1 must be a positive integer"
}

[[ -n "$MODEL" ]] || fail "MODEL must name a local Lite checkpoint directory"
command -v "$PYTHON" >/dev/null 2>&1 || fail "PYTHON executable not found: $PYTHON"

MODEL_PATH=$(
  "$PYTHON" - "$MODEL" <<'PYMODEL'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_dir():
    raise SystemExit(f"MODEL is not a directory: {path}")
try:
    config = json.loads((path / "config.json").read_text())
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid Lite config.json: {exc}") from exc
if config.get("architectures") != ["FLASHLocalForCausalLM"]:
    raise SystemExit("config.json must declare FLASHLocalForCausalLM")
if not any((path / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json")):
    raise SystemExit("checkpoint has no tokenizer metadata")
if not (path / "model.safetensors.index.json").is_file() and not any(
    path.glob("*.safetensors")
):
    raise SystemExit("checkpoint has no safetensors weights or index")
print(path.resolve())
PYMODEL
) || fail "checkpoint preflight failed"

positive_int WORLD_SIZE "$WORLD_SIZE"
[[ $WORLD_SIZE -eq 8 ]] || fail "WORLD_SIZE must be 8"
positive_int MAX_MODEL_LEN "$MAX_MODEL_LEN"
positive_int MAX_TOTAL_TOKENS "$MAX_TOTAL_TOKENS"
positive_int MAX_NUM_SEQS "$MAX_NUM_SEQS"
positive_int CHUNKED_PREFILL_SIZE "$CHUNKED_PREFILL_SIZE"
positive_int PREFIX_GRANULARITY "$PREFIX_GRANULARITY"
((MAX_MODEL_LEN <= 4096)) || fail "MAX_MODEL_LEN exceeds the bounded 4096-token admission"
((MAX_TOTAL_TOKENS <= 8192)) || fail "MAX_TOTAL_TOKENS exceeds the bounded 8192-token admission"
((MAX_NUM_SEQS <= 2)) || fail "MAX_NUM_SEQS exceeds the bounded BS2 admission"
((CHUNKED_PREFILL_SIZE <= 1024)) || fail "CHUNKED_PREFILL_SIZE exceeds the bounded 1024-token admission"
((CHUNKED_PREFILL_SIZE <= MAX_TOTAL_TOKENS)) || fail "CHUNKED_PREFILL_SIZE exceeds MAX_TOTAL_TOKENS"

IFS=',' read -r -a PREFILL_NPU_LIST <<< "$PREFILL_NPUS"
IFS=',' read -r -a DECODE_NPU_LIST <<< "$DECODE_NPUS"
[[ ${#PREFILL_NPU_LIST[@]} -eq 8 ]] || fail "PREFILL_NPUS must contain 8 device ids"
[[ ${#DECODE_NPU_LIST[@]} -eq 8 ]] || fail "DECODE_NPUS must contain 8 device ids"
declare -A DEVICE_IDS=()
for device in "${PREFILL_NPU_LIST[@]}"; do
  [[ $device =~ ^[0-9]+$ ]] || fail "invalid Prefill device id: $device"
  [[ -z ${DEVICE_IDS[$device]:-} ]] || fail "duplicate device id: $device"
  DEVICE_IDS[$device]=prefill
done
for device in "${DECODE_NPU_LIST[@]}"; do
  [[ $device =~ ^[0-9]+$ ]] || fail "invalid Decode device id: $device"
  [[ -z ${DEVICE_IDS[$device]:-} ]] || fail "P/D device sets overlap at id $device"
  DEVICE_IDS[$device]=decode
done

PORTS=(
  "$PREFILL_PORT"
  "$PREFILL_BOOTSTRAP_PORT"
  "$DECODE_PORT"
  "$PREFILL_DIST_PORT"
  "$DECODE_DIST_PORT"
  "$LB_PORT"
  "$PROMETHEUS_PORT"
)
for ((rank = 0; rank < WORLD_SIZE; rank++)); do
  PORTS+=(
    "$((PREFILL_HCCL_BASE_PORT + rank))"
    "$((DECODE_HCCL_BASE_PORT + rank))"
  )
done
declare -A PORT_SET=()
for port in "${PORTS[@]}"; do
  positive_int port "$port"
  ((port <= 65535)) || fail "port is out of range: $port"
  [[ -z ${PORT_SET[$port]:-} ]] || fail "ports must be distinct: $port"
  PORT_SET[$port]=1
done
PREFILL_WORKER_PORT_MIN=$((PREFILL_PORT + 100))
PREFILL_WORKER_PORT_MAX=$((PREFILL_PORT + 1000))
DECODE_WORKER_PORT_MIN=$((DECODE_PORT + 100))
DECODE_WORKER_PORT_MAX=$((DECODE_PORT + 1000))
((PREFILL_WORKER_PORT_MAX <= 65535)) || fail "Prefill worker port allocation window is out of range"
((DECODE_WORKER_PORT_MAX <= 65535)) || fail "Decode worker port allocation window is out of range"
if ((PREFILL_WORKER_PORT_MIN <= DECODE_WORKER_PORT_MAX && DECODE_WORKER_PORT_MIN <= PREFILL_WORKER_PORT_MAX)); then
  fail "P/D worker port allocation windows overlap"
fi
for port in "${PORTS[@]}"; do
  if ((port >= PREFILL_WORKER_PORT_MIN && port <= PREFILL_WORKER_PORT_MAX)); then
    fail "fixed port overlaps Prefill worker port allocation window: $port"
  fi
  if ((port >= DECODE_WORKER_PORT_MIN && port <= DECODE_WORKER_PORT_MAX)); then
    fail "fixed port overlaps Decode worker port allocation window: $port"
  fi
done
PORT_BINDINGS=(
  prefill 127.0.0.1 "$PREFILL_PORT"
  prefill-bootstrap 127.0.0.1 "$PREFILL_BOOTSTRAP_PORT"
  decode 127.0.0.1 "$DECODE_PORT"
  prefill-rendezvous 127.0.0.1 "$PREFILL_DIST_PORT"
  decode-rendezvous 127.0.0.1 "$DECODE_DIST_PORT"
  gateway "$LB_HOST" "$LB_PORT"
  metrics 0.0.0.0 "$PROMETHEUS_PORT"
)
for ((rank = 0; rank < WORLD_SIZE; rank++)); do
  PORT_BINDINGS+=(
    "prefill-hccl-rank-$rank" 0.0.0.0 "$((PREFILL_HCCL_BASE_PORT + rank))"
    "decode-hccl-rank-$rank" 0.0.0.0 "$((DECODE_HCCL_BASE_PORT + rank))"
  )
done
"$PYTHON" - "${PORT_BINDINGS[@]}" <<'PYPORTS' || fail "one or more requested ports are unavailable"
import socket
import sys

if (len(sys.argv) - 1) % 3:
    raise SystemExit("port bindings must be role/host/port triplets")

sockets = []
try:
    for index in range(1, len(sys.argv), 3):
        role, host, raw_port = sys.argv[index : index + 3]
        sock = socket.socket()
        try:
            sock.bind((host, int(raw_port)))
        except OSError as error:
            raise SystemExit(
                f"{role} listener unavailable at {host}:{raw_port}: {error}"
            ) from error
        sockets.append(sock)
finally:
    for sock in sockets:
        sock.close()
PYPORTS

COMMON_ARGS=(
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 127.0.0.1
  --device npu
  --dtype bfloat16
  --kv-cache-dtype auto
  --world-size 8
  --attn-tp-size 8
  --linear-attn-tp-size 8
  --dense-tp-size 8
  --ep-size 8
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-total-tokens "$MAX_TOTAL_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --prefix-granularity "$PREFIX_GRANULARITY"
  --attention-backend mla
  --kda-backend auto
  --sampling-backend greedy
  --enforce-eager
  --disable-prefill-graph
  --disable-pdl
  --disable-overlap-schedule
  --disable-autotune
  --disable-kvstore
  --disaggregation-transfer-backend mooncake
  --disaggregation-layerwise-interval 0
)
PREFILL_CMD=(
  "$PYTHON" -m smg_grpc_servicer.tokenspeed
  "${COMMON_ARGS[@]}"
  --port "$PREFILL_PORT"
  --dist-init-addr "127.0.0.1:$PREFILL_DIST_PORT"
  --disaggregation-bootstrap-port "$PREFILL_BOOTSTRAP_PORT"
  --disaggregation-mode prefill
)
DECODE_CMD=(
  "$PYTHON" -m smg_grpc_servicer.tokenspeed
  "${COMMON_ARGS[@]}"
  --port "$DECODE_PORT"
  --dist-init-addr "127.0.0.1:$DECODE_DIST_PORT"
  --disaggregation-mode decode
)
GATEWAY_CMD=(
  "$PYTHON" -m smg launch
  --pd-disaggregation
  --prefill "grpc://127.0.0.1:$PREFILL_PORT" "$PREFILL_BOOTSTRAP_PORT"
  --decode "grpc://127.0.0.1:$DECODE_PORT"
  --host "$LB_HOST"
  --port "$LB_PORT"
  --model-path "$MODEL_PATH"
  --tokenizer-path "$MODEL_PATH"
  --reasoning-parser passthrough
  --prefill-policy round_robin
  --decode-policy round_robin
  --max-concurrent-requests 2
  --queue-size 8
  --queue-timeout-secs 600
  --request-timeout-secs 600
  --log-level info
  --disable-retries
  --disable-load-monitoring
  --disable-circuit-breaker
  --disable-health-check
  --prometheus-port "$PROMETHEUS_PORT"
)

print_command() {
  local prefix=$1
  shift
  printf '%s' "$prefix"
  printf ' %q' "$@"
  printf '\n'
}

if ((CHECK_ONLY)); then
  print_command "[lite-npu-pd] prefill: ASCEND_RT_VISIBLE_DEVICES=$PREFILL_NPUS HCCL_IF_BASE_PORT=$PREFILL_HCCL_BASE_PORT" "${PREFILL_CMD[@]}"
  print_command "[lite-npu-pd] decode: ASCEND_RT_VISIBLE_DEVICES=$DECODE_NPUS HCCL_IF_BASE_PORT=$DECODE_HCCL_BASE_PORT" "${DECODE_CMD[@]}"
  print_command "[lite-npu-pd] gateway:" "${GATEWAY_CMD[@]}"
  exit 0
fi

mkdir -p "$LOG_DIR"
HCCN_CONF_PATH="$LOG_DIR/hccn.conf"
"$PYTHON" - "$HCCN_CONF_PATH" "$LOG_DIR/ascend-direct.env" \
  "$((WORLD_SIZE * 2))" "$PREFILL_HCCL_BASE_PORT" "$DECODE_HCCL_BASE_PORT" <<'PYHCCN' || \
  fail "Ascend HCCN preflight failed"
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
device_count = int(sys.argv[3])
tool = shutil.which("hccn_tool")
driver_tool = Path("/usr/local/Ascend/driver/tools/hccn_tool")
if tool is None and driver_tool.is_file() and os.access(driver_tool, os.X_OK):
    tool = str(driver_tool)
if tool is None:
    raise SystemExit("hccn_tool is unavailable")

lines = []
for device in range(device_count):
    result = subprocess.run(
        [tool, "-i", str(device), "-ip", "-g"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode:
        raise SystemExit(f"hccn_tool failed for device {device}")
    match = re.search(r"(?m)^\s*ipaddr\s*:\s*(\S+)\s*$", result.stdout)
    if match is None:
        raise SystemExit(f"hccn_tool returned no address for device {device}")
    lines.append(f"address_{device}={match.group(1)}")

payload = ("\n".join(lines) + "\n").encode()
temporary = output_path.with_suffix(output_path.suffix + ".tmp")
temporary.write_bytes(payload)
temporary.replace(output_path)
manifest_path.write_text(
    "\n".join(
        (
            f"hccn_sha256={hashlib.sha256(payload).hexdigest()}",
            f"prefill_hccl_base={sys.argv[4]}",
            f"decode_hccl_base={sys.argv[5]}",
        )
    )
    + "\n"
)
PYHCCN

"$PYTHON" - <<'PYIMPORTS' || fail "runtime import preflight failed"
import importlib

for module in (
    "torch",
    "torch_npu",
    "tokenspeed",
    "tokenspeed_kernel",
    "tokenspeed_kernel_npu",
    "smg",
    "smg_grpc_proto",
    "smg_grpc_servicer.tokenspeed.server",
):
    importlib.import_module(module)
PYIMPORTS
"$PYTHON" - <<'PYMOONCAKE' || fail "Mooncake native ABI preflight failed"
import os

from mooncake.engine import TransferEngine

required = (
    "initialize",
    "get_rpc_port",
    "register_memory",
    "unregister_memory",
    "transfer_sync_write",
    "batch_transfer_sync_write",
    "transfer_submit_write",
    "transfer_check_status",
)
missing = [name for name in required if not hasattr(TransferEngine, name)]
if missing:
    raise SystemExit(f"Mooncake TransferEngine is missing methods: {', '.join(missing)}")
os._exit(0)
PYMOONCAKE
env -u ASCEND_RT_VISIBLE_DEVICES "$PYTHON" - "${!DEVICE_IDS[@]}" <<'PYNPU' || fail "NPU visibility preflight failed"
import sys

import torch
import torch_npu  # noqa: F401

requested = [int(value) for value in sys.argv[1:]]
count = torch.npu.device_count()
if not requested or max(requested) >= count:
    raise SystemExit(f"requested physical NPU id exceeds device_count={count}: {requested}")
PYNPU

export NO_PROXY=${NO_PROXY:-127.0.0.1,localhost}
export no_proxy=${no_proxy:-127.0.0.1,localhost}
export TOKENSPEED_SKIP_GRPC_WARMUP=${TOKENSPEED_SKIP_GRPC_WARMUP:-1}

pids=()
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if ((${#pids[@]})); then
    stop_worker_pids "lite-npu-pd" "${WORKER_SHUTDOWN_TIMEOUT:-30}" "${pids[@]}"
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

wait_serving() {
  local role=$1
  local pid=$2
  local timeout=${3:-2400}
  local log="$LOG_DIR/${role}.log"
  local start=$SECONDS
  until grep -q "health status -> SERVING" "$log" 2>/dev/null; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[lite-npu-pd] $role exited before reaching SERVING (log=$log)" >&2
      tail -n 200 "$log" >&2 || true
      return 1
    fi
    if ((SECONDS - start > timeout)); then
      echo "[lite-npu-pd] timed out waiting for $role (log=$log)" >&2
      tail -n 200 "$log" >&2 || true
      return 1
    fi
    sleep 5
  done
}

wait_gateway() {
  local pid=$1
  local timeout=${2:-600}
  local start=$SECONDS
  until "$PYTHON" - "$LB_PORT" <<'PYREADY' >/dev/null 2>&1
import json
import sys
import urllib.request

with urllib.request.urlopen(
    f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=5
) as response:
    payload = json.load(response)
if "data" not in payload:
    raise SystemExit(1)
PYREADY
  do
    kill -0 "$pid" 2>/dev/null || return 1
    ((SECONDS - start <= timeout)) || return 1
    sleep 5
  done
}

(
  export ASCEND_RT_VISIBLE_DEVICES="$PREFILL_NPUS"
  export ASCEND_AUTO_CONNECT=1
  export HCCN_CONF_FILE="$HCCN_CONF_PATH"
  export HCCL_IF_BASE_PORT="$PREFILL_HCCL_BASE_PORT"
  export HCCL_INTRA_ROCE_ENABLE=1
  export HCCL_INTRA_PCIE_ENABLE=0
  export HCCL_CONNECT_TIMEOUT=600
  export HCCL_RDMA_TIMEOUT=20
  exec "${PREFILL_CMD[@]}"
) >"$LOG_DIR/prefill.log" 2>&1 &
pids+=("$!")
(
  export ASCEND_RT_VISIBLE_DEVICES="$DECODE_NPUS"
  export ASCEND_AUTO_CONNECT=1
  export HCCN_CONF_FILE="$HCCN_CONF_PATH"
  export HCCL_IF_BASE_PORT="$DECODE_HCCL_BASE_PORT"
  export HCCL_INTRA_ROCE_ENABLE=1
  export HCCL_INTRA_PCIE_ENABLE=0
  export HCCL_CONNECT_TIMEOUT=600
  export HCCL_RDMA_TIMEOUT=20
  exec "${DECODE_CMD[@]}"
) >"$LOG_DIR/decode.log" 2>&1 &
pids+=("$!")

wait_serving prefill "${pids[0]}"
wait_serving decode "${pids[1]}"
"${GATEWAY_CMD[@]}" >"$LOG_DIR/gateway.log" 2>&1 &
pids+=("$!")
wait_gateway "${pids[2]}" || fail "gateway did not return a TokenSpeed model list"

echo "[lite-npu-pd] serving on http://127.0.0.1:$LB_PORT/v1"
wait -n "${pids[@]}"
