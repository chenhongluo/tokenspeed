import json
import os
import socket
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("serve_lite_npu_pd_8p8d.sh")


def _checkpoint(path: Path, architecture: str = "FLASHLocalForCausalLM") -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"architectures": [architecture]}))
    (path / "tokenizer_config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"")
    return path


def _ports() -> list[int]:
    for base in range(20000, 56000):
        ports = [
            base,
            base + 1652,
            base + 2001,
            base + 1233,
            base + 3234,
            base - 1,
            base + 76,
        ]
        sockets = []
        try:
            for port in ports:
                sock = socket.socket()
                sock.bind(("0.0.0.0", port))
                sockets.append(sock)
        except OSError:
            continue
        finally:
            for sock in sockets:
                sock.close()
        return ports
    raise RuntimeError("no isolated worker port layout is available")


def _port_block(width: int, excluded: set[int]) -> int:
    for base in range(20000, 60000 - width):
        if any(base + offset in excluded for offset in range(width)):
            continue
        sockets = []
        try:
            for offset in range(width):
                sock = socket.socket()
                sock.bind(("0.0.0.0", base + offset))
                sockets.append(sock)
        except OSError:
            continue
        finally:
            for sock in sockets:
                sock.close()
        return base
    raise RuntimeError("no contiguous port block is available")


def _env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    ports = _ports()
    excluded = set(ports)
    excluded.update(range(ports[0] + 100, ports[0] + 1001))
    excluded.update(range(ports[2] + 100, ports[2] + 1001))
    prefill_hccl = _port_block(8, excluded)
    decode_hccl = _port_block(8, excluded | set(range(prefill_hccl, prefill_hccl + 8)))
    env.update(
        MODEL=str(_checkpoint(tmp_path / "model")),
        LITE_PD_LOG_DIR=str(tmp_path / "logs"),
        PREFILL_PORT=str(ports[0]),
        PREFILL_BOOTSTRAP_PORT=str(ports[1]),
        DECODE_PORT=str(ports[2]),
        PREFILL_DIST_PORT=str(ports[3]),
        DECODE_DIST_PORT=str(ports[4]),
        LB_PORT=str(ports[5]),
        PROMETHEUS_PORT=str(ports[6]),
        PREFILL_HCCL_BASE_PORT=str(prefill_hccl),
        DECODE_HCCL_BASE_PORT=str(decode_hccl),
    )
    return env


def _check(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", SCRIPT, "--check"],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )


def test_check_prints_the_bounded_8p8d_commands_without_side_effects(tmp_path):
    subprocess.run(["bash", "-n", SCRIPT], check=True)
    env = _env(tmp_path)

    result = _check(env)

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 3
    prefill, decode, gateway = lines
    assert "ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in prefill
    assert "ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15" in decode
    assert f"HCCL_IF_BASE_PORT={env['PREFILL_HCCL_BASE_PORT']}" in prefill
    assert f"HCCL_IF_BASE_PORT={env['DECODE_HCCL_BASE_PORT']}" in decode
    for command in (prefill, decode):
        for flag in (
            "--device npu",
            "--world-size 8",
            "--attn-tp-size 8",
            "--linear-attn-tp-size 8",
            "--dense-tp-size 8",
            "--ep-size 8",
            "--max-model-len 4096",
            "--max-total-tokens 8192",
            "--max-num-seqs 2",
            "--chunked-prefill-size 1024",
            "--prefix-granularity 64",
            "--attention-backend mla",
            "--sampling-backend greedy",
            "--disable-prefill-graph",
            "--disable-pdl",
        ):
            assert flag in command
    assert "--enforce-eager" in prefill
    assert "--no-enable-prefix-caching" in prefill
    assert "--disable-overlap-schedule" in prefill
    assert "--cudagraph-capture-sizes" not in prefill
    assert "--max-cudagraph-capture-size" not in prefill
    assert "--enforce-eager" not in decode
    assert "--no-enable-prefix-caching" not in decode
    assert "--disable-overlap-schedule" not in decode
    assert "--cudagraph-capture-sizes 1 2" in decode
    assert "--max-cudagraph-capture-size 2" in decode
    assert "--disaggregation-mode prefill" in prefill
    assert "--disaggregation-mode decode" in decode
    assert "--pd-disaggregation" in gateway
    assert "--prefill grpc://127.0.0.1:" in gateway
    assert "--decode grpc://127.0.0.1:" in gateway
    assert not Path(env["LITE_PD_LOG_DIR"]).exists()


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"PREFILL_NPUS": "0,1,2,3,4,5,6"}, "must contain 8 device ids"),
        ({"DECODE_NPUS": "7,8,9,10,11,12,13,14"}, "device sets overlap"),
        ({"MAX_MODEL_LEN": "4097"}, "exceeds the bounded 4096-token"),
        ({"MAX_TOTAL_TOKENS": "0"}, "must be a positive integer"),
    ),
)
def test_check_rejects_invalid_topology_or_capacity(tmp_path, override, message):
    env = _env(tmp_path)
    env.update(override)

    result = _check(env)

    assert result.returncode == 2
    assert message in result.stderr
    assert not Path(env["LITE_PD_LOG_DIR"]).exists()


@pytest.mark.parametrize("port_name", ("PREFILL_BOOTSTRAP_PORT", "PREFILL_DIST_PORT"))
def test_check_rejects_fixed_port_in_worker_allocation_window(tmp_path, port_name):
    env = _env(tmp_path)
    env[port_name] = str(int(env["PREFILL_PORT"]) + 652)

    result = _check(env)

    assert result.returncode == 2
    assert "fixed port overlaps Prefill worker port allocation window" in result.stderr


def test_check_rejects_overlapping_worker_allocation_windows(tmp_path):
    env = _env(tmp_path)
    env["DECODE_PORT"] = str(int(env["PREFILL_PORT"]) + 99)

    result = _check(env)

    assert result.returncode == 2
    assert "P/D worker port allocation windows overlap" in result.stderr


def test_check_rejects_invalid_checkpoint_and_duplicate_ports(tmp_path):
    env = _env(tmp_path)
    del env["MODEL"]
    result = _check(env)
    assert result.returncode == 2
    assert "MODEL must name a local Lite checkpoint" in result.stderr

    env = _env(tmp_path / "invalid")
    Path(env["MODEL"], "config.json").write_text(
        json.dumps({"architectures": ["OtherForCausalLM"]})
    )
    result = _check(env)
    assert result.returncode == 2
    assert "FLASHLocalForCausalLM" in result.stderr

    env = _env(tmp_path / "second")
    env["DECODE_PORT"] = env["PREFILL_PORT"]
    result = _check(env)
    assert result.returncode == 2
    assert "ports must be distinct" in result.stderr

    env = _env(tmp_path / "third")
    env["DECODE_HCCL_BASE_PORT"] = env["PREFILL_HCCL_BASE_PORT"]
    result = _check(env)
    assert result.returncode == 2
    assert "ports must be distinct" in result.stderr


def test_check_rejects_metrics_port_occupied_on_wildcard_address(tmp_path):
    with socket.socket() as occupied:
        occupied.bind(("0.0.0.0", 0))
        env = _env(tmp_path)
        env["PROMETHEUS_PORT"] = str(occupied.getsockname()[1])

        result = _check(env)

    assert result.returncode == 2
    assert "metrics listener unavailable at 0.0.0.0" in result.stderr
    assert not Path(env["LITE_PD_LOG_DIR"]).exists()


def test_normal_mode_preflights_the_native_mooncake_abi():
    script = SCRIPT.read_text()

    assert "from mooncake.engine import TransferEngine" in script
    assert '"mooncake",' not in script
    for method in (
        "initialize",
        "get_rpc_port",
        "register_memory",
        "unregister_memory",
        "transfer_sync_write",
        "batch_transfer_sync_write",
        "transfer_submit_write",
        "transfer_check_status",
    ):
        assert f'"{method}"' in script
    assert "os._exit(0)" in script


def test_normal_mode_rejects_malformed_hccn_output(tmp_path):
    env = _env(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    tool = fake_bin / "hccn_tool"
    tool.write_text("#!/bin/sh\nprintf 'device has no address\\n'\n")
    tool.chmod(0o755)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        ["bash", SCRIPT],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "Ascend HCCN preflight failed" in result.stderr
    assert "hccn_tool returned no address for device 0" in result.stderr


def test_normal_mode_sets_isolated_ascend_direct_channel_env():
    script = SCRIPT.read_text()

    assert 'Path("/usr/local/Ascend/driver/tools/hccn_tool")' in script
    for assignment in (
        "ASCEND_AUTO_CONNECT=1",
        'HCCN_CONF_FILE="$HCCN_CONF_PATH"',
        "HCCL_INTRA_ROCE_ENABLE=1",
        "HCCL_INTRA_PCIE_ENABLE=0",
        "HCCL_CONNECT_TIMEOUT=600",
        "HCCL_RDMA_TIMEOUT=20",
    ):
        assert script.count(assignment) == 2
    assert 'HCCL_IF_BASE_PORT="$PREFILL_HCCL_BASE_PORT"' in script
    assert 'HCCL_IF_BASE_PORT="$DECODE_HCCL_BASE_PORT"' in script
