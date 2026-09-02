# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Exact-checkpoint mmap, PSS, HBM, and numerical probe for Lite OE."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from safetensors.torch import load_file

from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig
from tokenspeed.runtime.layers.over_embedding import (
    CheckpointedTailOEStatePreparer,
    HostLongCatOverEmbedding,
)
from tokenspeed.runtime.models.flash_local_checkpoint import (
    FLASHLocalCheckpointLayout,
)

EXACT_OE_PAYLOAD_BYTES = 28_991_102_976
_SMAPS_FIELDS = (
    "Size",
    "Rss",
    "Pss",
    "Shared_Clean",
    "Private_Clean",
    "Private_Dirty",
    "Anonymous",
)
_SMAPS_HEADER = re.compile(r"^([0-9a-f]+)-([0-9a-f]+)\s+(\S+)\s+")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _smaps_entries(pid: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    with open(f"/proc/{pid}/smaps") as source:
        for line in source:
            match = _SMAPS_HEADER.match(line)
            if match:
                current = {
                    "start": int(match.group(1), 16),
                    "end": int(match.group(2), 16),
                    "file_backed": len(line.split(maxsplit=5)) == 6
                    and not line.split(maxsplit=5)[5].startswith("["),
                    **{field: 0 for field in _SMAPS_FIELDS},
                }
                entries.append(current)
                continue
            if current is None or ":" not in line:
                continue
            field, raw_value = line.split(":", 1)
            if field in _SMAPS_FIELDS:
                current[field] = int(raw_value.split()[0])
    return entries


def mapping_metrics(pid: int, pointers: list[int]) -> dict[str, Any]:
    """Aggregate each unique mapping containing a supplied tensor pointer."""
    entries = _smaps_entries(pid)
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for pointer in pointers:
        entry = next(
            (
                candidate
                for candidate in entries
                if candidate["start"] <= pointer < candidate["end"]
            ),
            None,
        )
        if entry is None:
            raise RuntimeError("A Lite OE tensor pointer has no /proc mapping.")
        selected[(entry["start"], entry["end"])] = entry
    return {
        "mapping_count": len(selected),
        "all_file_backed": all(entry["file_backed"] for entry in selected.values()),
        **{
            f"{field.lower()}_kib": sum(entry[field] for entry in selected.values())
            for field in _SMAPS_FIELDS
        },
    }


def rollup_metrics(pid: int) -> dict[str, int]:
    result = {f"{field.lower()}_kib": 0 for field in _SMAPS_FIELDS}
    with open(f"/proc/{pid}/smaps_rollup") as source:
        for line in source:
            if ":" not in line:
                continue
            field, raw_value = line.split(":", 1)
            if field in _SMAPS_FIELDS:
                result[f"{field.lower()}_kib"] = int(raw_value.split()[0])
    return result


def _normalize_file_mappings(pointers: list[int]) -> None:
    """Drop incidental PTEs and disable read-ahead for the measured mappings."""
    entries = _smaps_entries(os.getpid())
    selected = {
        (entry["start"], entry["end"]): entry
        for pointer in pointers
        for entry in entries
        if entry["start"] <= pointer < entry["end"]
    }
    if not selected or not all(entry["file_backed"] for entry in selected.values()):
        raise RuntimeError("Lite OE PTE normalization requires file-backed mappings.")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    for entry in selected.values():
        for advice in (4, 1):  # MADV_DONTNEED, then MADV_RANDOM.
            if libc.madvise(entry["start"], entry["end"] - entry["start"], advice):
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))


def _oe_names(config: FLASHLocalConfig) -> tuple[list[str], list[str]]:
    layout = FLASHLocalCheckpointLayout(config)
    table_names = []
    projection_names = []
    for table_id in range(config.oe_component_count):
        table_name = f"model.ngram_embeddings.embedders.{table_id}.weight"
        projection_name = f"model.ngram_embeddings.post_projs.{table_id}.weight"
        if layout.spec(table_name).category != "host-oe":
            raise RuntimeError("Lite checkpoint layout lost host OE placement.")
        table_names.append(table_name)
        projection_names.append(projection_name)
    return table_names, projection_names


def _load_selected(
    checkpoint: Path, names: list[str]
) -> tuple[dict[str, torch.Tensor], int]:
    index = _read_json(checkpoint / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Safetensors index is missing weight_map.")
    selected: dict[str, torch.Tensor] = {}
    shards: dict[str, list[str]] = {}
    for name in names:
        shard = weight_map.get(name)
        if not isinstance(shard, str):
            raise ValueError(f"Safetensors index is missing Lite OE key {name!r}.")
        shard_path = PurePosixPath(shard)
        if shard_path.is_absolute() or ".." in shard_path.parts:
            raise ValueError("Safetensors index contains an unsafe shard path.")
        shards.setdefault(shard, []).append(name)
    for shard, shard_names in shards.items():
        loaded = load_file(checkpoint / shard, device="cpu")
        for name in shard_names:
            selected[name] = loaded[name]
        del loaded
    return selected, len(shards)


def load_oe_checkpoint(
    checkpoint: Path,
    *,
    device: int | None,
    after_table_adoption: Callable[[], None] | None = None,
) -> tuple[HostLongCatOverEmbedding, dict[str, Any], torch.Tensor]:
    """Load only Lite OE tensors and retain table safetensors mappings."""
    config = FLASHLocalConfig.from_dict(_read_json(checkpoint / "config.json"))
    table_names, projection_names = _oe_names(config)
    selected, shard_count = _load_selected(checkpoint, table_names + projection_names)
    layer = HostLongCatOverEmbedding(config)
    table_pointers = []
    payload_bytes = 0
    for table_id, name in enumerate(table_names):
        source = selected[name]
        expected = (
            config.oe_table_rows(table_id),
            config.oe_hidden_size,
        )
        if tuple(source.shape) != expected or source.dtype != torch.bfloat16:
            raise ValueError(f"Lite OE table {table_id} has an invalid shape or dtype.")
        layer.embedders[table_id].weight.data = source
        if (
            layer.embedders[table_id].weight.untyped_storage().data_ptr()
            != source.untyped_storage().data_ptr()
        ):
            raise RuntimeError("Lite OE table storage adoption did not alias.")
        table_pointers.append(source.data_ptr())
        payload_bytes += source.numel() * source.element_size()
    if after_table_adoption is not None:
        after_table_adoption()
    for table_id, name in enumerate(projection_names):
        source = selected[name]
        expected = (config.hidden_size, config.oe_hidden_size)
        if tuple(source.shape) != expected or source.dtype != torch.bfloat16:
            raise ValueError(
                f"Lite OE projection {table_id} has an invalid shape or dtype."
            )
        layer.projection.data[table_id].copy_(source.t())
    packed_cpu = layer.projection.detach()
    if device is not None:
        layer.projection = torch.nn.Parameter(
            packed_cpu.to(f"npu:{device}"), requires_grad=False
        )
        layer.ignore_tokens = layer.ignore_tokens.to(f"npu:{device}")
    del selected
    gc.collect()
    for table_id, table in enumerate(layer.embedders):
        if not torch.equal(table.weight[0], table.weight[0].clone()):
            raise RuntimeError(f"Lite OE table {table_id} mapping is not readable.")
    return (
        layer,
        {
            "payload_bytes": payload_bytes,
            "projection_bytes": layer.projection.numel()
            * layer.projection.element_size(),
            "table_pointers": table_pointers,
            "shard_count": shard_count,
        },
        packed_cpu,
    )


def _touch_tables(layer: HostLongCatOverEmbedding, touch_mib: int) -> float:
    total_pages = max(1, touch_mib * 256)
    pages_per_table = max(1, total_pages // len(layer.embedders))
    checksum = 0.0
    for table in layer.embedders:
        flat = table.weight.view(-1)
        elements_per_page = os.sysconf("SC_PAGE_SIZE") // flat.element_size()
        pages = min(pages_per_table, math.ceil(flat.numel() / elements_per_page))
        indices = torch.arange(pages) * elements_per_page
        checksum += float(flat[indices].float().sum())
    return checksum


def _npu_snapshot(device: int) -> dict[str, int]:
    torch.npu.synchronize(device)
    return {
        "allocated_bytes": int(torch.npu.memory_allocated(device)),
        "reserved_bytes": int(torch.npu.memory_reserved(device)),
        "max_allocated_bytes": int(torch.npu.max_memory_allocated(device)),
    }


def _initialize_npu(device: int) -> dict[str, int]:
    __import__("torch_npu")
    torch.npu.set_device(device)
    temporary = torch.empty(1, device=f"npu:{device}")
    del temporary
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats(device)
    return _npu_snapshot(device)


def _numerical_probe(
    layer: HostLongCatOverEmbedding, packed_cpu: torch.Tensor, device: int
) -> dict[str, Any]:
    config = layer.config
    cases = []
    for token_count in (1, 2, 4, 32, 128, 1024):
        tokens = (torch.arange(token_count, dtype=torch.int64) * 17 + 5).remainder(
            config.vocab_size
        )
        if token_count > 1:
            tokens[0] = config.eos_token_id
        if token_count > 2:
            tokens[1] = config.special_token_ids[0]
        context = torch.full((1, 3), config.eos_token_id, dtype=torch.int64)
        ids, _, _ = layer.ngram_ids(tokens, context, [token_count])
        raw = layer.lookup_host(ids)
        word = torch.randn(
            (token_count, config.hidden_size),
            generator=torch.Generator().manual_seed(2027 + token_count),
            dtype=torch.float32,
        ).to(torch.bfloat16)
        projected = (
            raw.reshape(token_count, config.hidden_size).float()
            @ packed_cpu[:].reshape(config.hidden_size, config.hidden_size).float()
        )
        expected = (word.float() + projected) / math.sqrt(config.oe_component_count + 1)
        special = torch.tensor(
            [int(token) in config.special_token_ids for token in tokens]
        )
        expected[special] = word[special].float()
        actual = layer.project_and_merge(
            word.to(f"npu:{device}"),
            raw.to(f"npu:{device}"),
            tokens.to(f"npu:{device}"),
        )
        torch.npu.synchronize(device)
        actual_cpu = actual.float().cpu()
        delta = actual_cpu - expected
        relative_l2 = float(
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(expected).clamp_min(1e-12)
        )
        passed = bool(torch.allclose(actual_cpu, expected, atol=0.02, rtol=0.02))
        passed &= bool(torch.isfinite(actual_cpu).all())
        passed &= bool(torch.equal(actual_cpu[special], word[special].float()))
        cases.append(
            {
                "path": "decode" if token_count <= 32 else "prefill",
                "tokens": token_count,
                "passed": passed,
                "max_abs": float(delta.abs().max()),
                "relative_l2": relative_l2,
            }
        )

    initial = torch.full((2, 3), config.eos_token_id, dtype=torch.int64)
    one_shot_ids, _, one_shot_tail = layer.ngram_ids(
        torch.tensor([5, 6, 7, 8, 9, 10, 11]), initial, [4, 3]
    )
    first_ids, _, first_tail = layer.ngram_ids(torch.tensor([5, 6, 9]), initial, [2, 1])
    second_ids, _, second_tail = layer.ngram_ids(
        torch.tensor([7, 8, 10, 11]), first_tail, [2, 2]
    )
    continued_ids = torch.cat(
        (first_ids[:2], second_ids[:2], first_ids[2:], second_ids[2:])
    )
    continuation_passed = bool(torch.equal(continued_ids, one_shot_ids)) and bool(
        torch.equal(second_tail, one_shot_tail)
    )

    staging = torch.zeros(
        (2, config.oe_component_count, config.oe_hidden_size),
        dtype=torch.bfloat16,
        device=f"npu:{device}",
    )
    word = torch.zeros(
        (2, config.hidden_size), dtype=torch.bfloat16, device=f"npu:{device}"
    )
    tokens = torch.tensor([5, 6], dtype=torch.int64, device=f"npu:{device}")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        graph_output = layer.project_and_merge(word, staging, tokens)
    graph.replay()
    torch.npu.synchronize(device)
    first = graph_output.clone()
    staging.copy_(
        torch.arange(staging.numel(), device=f"npu:{device}").reshape_as(staging)
    )
    word.fill_(0.125)
    tokens.copy_(torch.tensor([7, 1], dtype=torch.int64, device=f"npu:{device}"))
    graph.replay()
    torch.npu.synchronize(device)
    eager = layer.project_and_merge(word, staging, tokens)
    graph_passed = bool(torch.equal(graph_output, eager)) and bool(
        torch.isfinite(graph_output).all()
    )
    graph_passed &= not bool(torch.equal(first, graph_output))
    return {
        "cases": cases,
        "two_request_continuation_passed": continuation_passed,
        "graph_bs2_passed": graph_passed,
    }


def run_worker(args: argparse.Namespace) -> None:
    device = args.device if args.device >= 0 else None
    hbm: dict[str, dict[str, int]] = {}
    if device is not None:
        hbm["baseline"] = _initialize_npu(device)
    baseline_rollup = rollup_metrics(os.getpid())

    def record_table_hbm() -> None:
        assert device is not None
        hbm["table_adopted"] = _npu_snapshot(device)

    layer, storage, packed_cpu = load_oe_checkpoint(
        Path(args.checkpoint),
        device=device,
        after_table_adoption=record_table_hbm if device is not None else None,
    )
    if device is not None:
        hbm["projection_loaded"] = _npu_snapshot(device)
    pointers = storage["table_pointers"]
    storage["mapping_rss_before_normalize_kib"] = mapping_metrics(
        os.getpid(), pointers
    )["rss_kib"]
    _normalize_file_mappings(pointers)
    storage["mapping_rss_after_normalize_kib"] = mapping_metrics(os.getpid(), pointers)[
        "rss_kib"
    ]
    checksum = _touch_tables(layer, args.touch_mib)
    numerics = (
        _numerical_probe(layer, packed_cpu, device)
        if args.exact_numerics and device is not None
        else None
    )
    if device is not None:
        bucket = 1 if args.role == "prefill" else 2
        preparer = CheckpointedTailOEStatePreparer(
            layer,
            context_pages=torch.zeros(
                (3, 3), dtype=torch.int32, device=f"npu:{device}"
            ),
            checkpoint_granularity=128,
            max_request_slots=2,
            max_graph_tokens=bucket,
            device=f"npu:{device}",
        )
        preparer.prepare(
            request_ids=["probe"],
            request_pool_indices=[0],
            input_ids=torch.tensor([5]),
            lengths=[1],
            before_lengths=[0],
            block_table=torch.tensor([[1]], dtype=torch.int32),
            graph_tokens=bucket if args.role == "decode" else None,
        )
        hbm["staging_published"] = _npu_snapshot(device)
    current_rollup = rollup_metrics(os.getpid())
    _write_json(
        Path(args.ready),
        {
            "pid": os.getpid(),
            "rank": args.rank,
            "role": args.role,
            "device": device,
            "storage": storage,
            "checksum_finite": math.isfinite(checksum),
            "anonymous_delta_kib": current_rollup["anonymous_kib"]
            - baseline_rollup["anonymous_kib"],
            "hbm": hbm,
            "numerics": numerics,
        },
    )
    release = Path(args.release)
    deadline = time.monotonic() + args.timeout
    while not release.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Lite OE probe worker timed out at the ledger barrier.")
        time.sleep(0.05)


def _role(rank: int, workers: int) -> str:
    return "prefill" if rank < (workers + 1) // 2 else "decode"


def run_ledger(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers <= 0:
        raise ValueError("workers must be positive.")
    devices = (
        [] if not args.devices else [int(value) for value in args.devices.split(",")]
    )
    if devices and len(devices) != args.workers:
        raise ValueError("devices must provide one NPU ID per worker.")
    with tempfile.TemporaryDirectory(prefix="lite-oe-ledger-") as directory:
        root = Path(directory)
        release = root / "release"
        processes: list[tuple[subprocess.Popen[str], Any]] = []
        ready_paths = []
        try:
            for rank in range(args.workers):
                ready = root / f"ready-{rank}.json"
                log = open(root / f"worker-{rank}.log", "w+")
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--checkpoint",
                    args.checkpoint,
                    "--ready",
                    str(ready),
                    "--release",
                    str(release),
                    "--rank",
                    str(rank),
                    "--role",
                    _role(rank, args.workers),
                    "--device",
                    str(devices[rank] if devices else -1),
                    "--touch-mib",
                    str(args.touch_mib),
                    "--timeout",
                    str(args.timeout),
                ]
                if args.exact_numerics and rank == 0:
                    command.append("--exact-numerics")
                processes.append(
                    (subprocess.Popen(command, stdout=log, stderr=log, text=True), log)
                )
                ready_paths.append(ready)

            deadline = time.monotonic() + args.timeout
            while not all(path.exists() for path in ready_paths):
                for process, log in processes:
                    if process.poll() is not None:
                        log.flush()
                        log.seek(0)
                        raise RuntimeError(
                            "Lite OE probe worker exited before the ledger barrier: "
                            + log.read()[-4000:]
                        )
                if time.monotonic() >= deadline:
                    raise TimeoutError("Lite OE ledger timed out waiting for workers.")
                time.sleep(0.1)

            workers = [_read_json(path) for path in ready_paths]
            for worker in workers:
                worker["checkpoint_mapping"] = mapping_metrics(
                    worker["pid"], worker["storage"].pop("table_pointers")
                )
                worker["process_rollup"] = rollup_metrics(worker["pid"])
            mappings = [worker["checkpoint_mapping"] for worker in workers]
            sum_pss = sum(mapping["pss_kib"] for mapping in mappings)
            max_rss = max(mapping["rss_kib"] for mapping in mappings)
            checks = {
                "payload_exact": all(
                    worker["storage"]["payload_bytes"]
                    == (
                        EXACT_OE_PAYLOAD_BYTES
                        if args.require_exact_layout
                        else workers[0]["storage"]["payload_bytes"]
                    )
                    for worker in workers
                ),
                "file_backed": all(mapping["all_file_backed"] for mapping in mappings),
                "mapping_anonymous_zero": all(
                    mapping["anonymous_kib"] == 0 for mapping in mappings
                ),
                "anonymous_delta_bounded": all(
                    worker["anonymous_delta_kib"] <= 512 * 1024 for worker in workers
                ),
                "shared_clean": args.workers == 1
                or all(mapping["shared_clean_kib"] > 0 for mapping in mappings),
                "pss_one_copy": args.workers == 1
                or sum_pss <= max_rss * 1.25 + 16 * 1024,
                "finite": all(worker["checksum_finite"] for worker in workers),
                "normalized_mapping_rss": all(
                    worker["storage"]["mapping_rss_after_normalize_kib"] <= 64
                    for worker in workers
                ),
            }
            if devices:
                checks["host_table_not_in_hbm"] = all(
                    worker["hbm"]["table_adopted"]["allocated_bytes"]
                    - worker["hbm"]["baseline"]["allocated_bytes"]
                    <= 16 * 1024 * 1024
                    for worker in workers
                )
                checks["projection_logical_bytes"] = all(
                    worker["storage"]["projection_bytes"] == 18 * 1024 * 1024
                    for worker in workers
                )
            if args.exact_numerics:
                numerical = workers[0]["numerics"]
                checks["numerics"] = numerical is not None and all(
                    case["passed"] for case in numerical["cases"]
                )
                checks["two_request_continuation"] = bool(
                    numerical and numerical["two_request_continuation_passed"]
                )
                checks["graph_bs2"] = bool(numerical and numerical["graph_bs2_passed"])
            result = {
                "schema_version": 1,
                "workers": workers,
                "totals": {
                    "checkpoint_pss_kib": sum_pss,
                    "checkpoint_rss_kib": sum(
                        mapping["rss_kib"] for mapping in mappings
                    ),
                    "max_checkpoint_rss_kib": max_rss,
                },
                "checks": checks,
                "passed": all(checks.values()),
            }
            return result
        finally:
            release.touch()
            for process, log in processes:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    ledger = subparsers.add_parser("ledger")
    ledger.add_argument("--checkpoint", required=True)
    ledger.add_argument("--output", required=True)
    ledger.add_argument("--workers", type=int, default=1)
    ledger.add_argument("--devices", default="")
    ledger.add_argument("--touch-mib", type=int, default=256)
    ledger.add_argument("--timeout", type=int, default=900)
    ledger.add_argument("--require-exact-layout", action="store_true")
    ledger.add_argument("--exact-numerics", action="store_true")

    worker = subparsers.add_parser("worker")
    worker.add_argument("--checkpoint", required=True)
    worker.add_argument("--ready", required=True)
    worker.add_argument("--release", required=True)
    worker.add_argument("--rank", type=int, required=True)
    worker.add_argument("--role", choices=("prefill", "decode"), required=True)
    worker.add_argument("--device", type=int, required=True)
    worker.add_argument("--touch-mib", type=int, required=True)
    worker.add_argument("--timeout", type=int, required=True)
    worker.add_argument("--exact-numerics", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "worker":
        run_worker(args)
        return
    result = run_ledger(args)
    _write_json(Path(args.output), result)
    if not result["passed"]:
        raise SystemExit("Lite OE checkpoint ledger failed a hard gate.")


if __name__ == "__main__":
    main()
