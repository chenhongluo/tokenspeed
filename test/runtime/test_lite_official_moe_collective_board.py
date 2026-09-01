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

"""Direct-board admission for Huawei's non-V2 MoE collectives.

Launch explicitly on eight idle NPUs::

    LITE_OFFICIAL_MOE_BOARD=1 HCCL_BUFFSIZE=128 \
      torchrun --standalone --nproc-per-node=8 \
      test/runtime/test_lite_official_moe_collective_board.py

This is an offline admission harness. It must not be imported by production.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

GROUPS = 4
EXPERTS = 384
IDENTITY_EXPERTS = 32
LOCAL_EXPERTS = 48
TOPK = 12
HIDDEN = 768
EP_SIZE = 8


def _legalize_identity_routes(
    expert_ids: torch.Tensor, expert_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace identity routes with distinct zero-weight real expert routes."""
    if expert_ids.ndim != 2 or expert_ids.shape[1] != TOPK:
        raise ValueError(f"expected expert_ids [BS,{TOPK}], got {expert_ids.shape}")
    if expert_weights.shape != expert_ids.shape:
        raise ValueError("expert_weights must match expert_ids")
    if not bool(((expert_ids >= 0) & (expert_ids < EXPERTS + IDENTITY_EXPERTS)).all()):
        raise ValueError("logical expert ID is out of range")

    legal_ids = expert_ids.clone().to(torch.int32)
    legal_weights = expert_weights.clone().to(torch.float32)
    identity_sum = torch.zeros(expert_ids.shape[0], dtype=torch.float32)
    for token in range(expert_ids.shape[0]):
        used = {int(value) for value in expert_ids[token] if int(value) < EXPERTS}
        replacement = 0
        for route in range(TOPK):
            if int(expert_ids[token, route]) < EXPERTS:
                continue
            identity_sum[token] += expert_weights[token, route].float()
            while replacement in used:
                replacement += 1
            legal_ids[token, route] = replacement
            legal_weights[token, route] = 0.0
            used.add(replacement)
            replacement += 1
        if len(set(legal_ids[token].tolist())) != TOPK:
            raise ValueError("legalized routes must be distinct per token")
    return legal_ids, legal_weights, identity_sum


def _logical_routes(batch: int, group: int, kind: str, shift: int = 0):
    token = torch.arange(batch, dtype=torch.int64)[:, None]
    route = torch.arange(TOPK, dtype=torch.int64)[None, :]
    if kind == "hot":
        expert_ids = (route + shift).expand(batch, -1).remainder(EXPERTS)
    else:
        expert_ids = (token * 37 + route * 17 + group * 53 + shift).remainder(EXPERTS)
    if kind == "mixed":
        identity = (route + token + group).remainder(2) == 0
        identity_ids = EXPERTS + (route + token).remainder(IDENTITY_EXPERTS)
        expert_ids = torch.where(identity, identity_ids, expert_ids)
    elif kind == "identity_only":
        expert_ids = EXPERTS + (route + token).remainder(IDENTITY_EXPERTS)
    elif kind not in {"real_only", "hot"}:
        raise ValueError(f"unknown route kind: {kind}")

    raw = torch.cos((token + 1) * (route + group + 1) * 0.03125)
    weights = torch.softmax(raw.float(), dim=-1) * 6.0
    return expert_ids.to(torch.int32), weights


def test_identity_routes_become_distinct_zero_weight_real_routes():
    for kind in ("real_only", "mixed", "identity_only", "hot"):
        logical_ids, logical_weights = _logical_routes(2, 1, kind)
        legal_ids, legal_weights, identity_sum = _legalize_identity_routes(
            logical_ids, logical_weights
        )
        assert bool(((legal_ids >= 0) & (legal_ids < EXPERTS)).all())
        assert all(len(set(row.tolist())) == TOPK for row in legal_ids)
        assert torch.equal(
            legal_weights,
            torch.where(logical_ids < EXPERTS, logical_weights, 0.0),
        )
        assert torch.allclose(
            legal_weights.sum(dim=-1) + identity_sum,
            logical_weights.sum(dim=-1),
        )


def _hccl_name(group, device: torch.device) -> str:
    import torch.distributed as dist

    backend = group._get_backend(device)
    return backend.get_hccl_comm_name(dist.get_rank(group))


def _official_pair(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    group_name: str,
    rank: int,
    *,
    graph: bool,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    import torch_npu

    dispatch = torch_npu.npu_moe_distribute_dispatch(
        x=hidden,
        expert_ids=expert_ids,
        group_ep=group_name,
        ep_world_size=EP_SIZE,
        ep_rank_id=rank,
        moe_expert_num=EXPERTS,
        group_tp=group_name if graph else "",
        tp_world_size=0,
        tp_rank_id=0,
        shared_expert_rank_num=0,
        quant_mode=0,
        global_bs=0,
        expert_token_nums_type=1,
    )
    combined = torch_npu.npu_moe_distribute_combine(
        expand_x=dispatch[0],
        expert_ids=expert_ids,
        expand_idx=dispatch[2],
        ep_send_counts=dispatch[4],
        expert_scales=expert_weights,
        group_ep=group_name,
        ep_world_size=EP_SIZE,
        ep_rank_id=rank,
        moe_expert_num=EXPERTS,
        expand_scales=dispatch[6],
        group_tp=group_name if graph else "",
        tp_world_size=0,
        tp_rank_id=0,
        shared_expert_rank_num=0,
        global_bs=0,
        comm_quant_mode=0,
        group_list_type=1,
    )
    return combined, list(dispatch)


def _inputs(batch: int, kind: str, shift: int, device: torch.device):
    generator = torch.Generator().manual_seed(8100 + batch + shift)
    hidden = torch.randn(
        batch, GROUPS, HIDDEN, dtype=torch.bfloat16, generator=generator
    )
    legal_ids, legal_weights, identity = [], [], []
    logical_weights = []
    for group in range(GROUPS):
        ids, weights = _logical_routes(batch, group, kind, shift)
        ids, real_weights, identity_sum = _legalize_identity_routes(ids, weights)
        legal_ids.append(ids)
        legal_weights.append(real_weights)
        identity.append(identity_sum)
        logical_weights.append(weights)
    return (
        hidden.to(device),
        torch.stack(legal_ids).to(device),
        torch.stack(legal_weights).to(device),
        torch.stack(identity).to(device),
        torch.stack(logical_weights).to(device),
    )


def _run_groups(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    identity_weights: torch.Tensor,
    group_name: str,
    rank: int,
    *,
    graph: bool,
):
    outputs = []
    dispatch_metadata = []
    for group in range(GROUPS):
        output, metadata = _official_pair(
            hidden[:, group],
            expert_ids[group],
            expert_weights[group],
            group_name,
            rank,
            graph=graph,
        )
        outputs.append(output + hidden[:, group] * identity_weights[group, :, None])
        dispatch_metadata.append(metadata)
    return torch.stack(outputs, dim=1), dispatch_metadata


def _assert_oracle(
    actual: torch.Tensor, hidden: torch.Tensor, logical_weights: torch.Tensor
) -> dict[str, float]:
    expected = hidden.float() * logical_weights.sum(dim=-1).permute(1, 0)[..., None]
    assert bool(torch.isfinite(actual).all())
    difference = actual.float() - expected
    rel_l2 = float(
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(expected).clamp_min(1e-12)
    )
    max_abs = float(difference.abs().max())
    assert torch.allclose(actual.float(), expected, atol=2e-2, rtol=2e-2)
    assert rel_l2 <= 5e-3
    return {"max_abs": max_abs, "rel_l2": rel_l2}


def _metadata_summary(metadata: list[list[torch.Tensor]]) -> list[dict]:
    return [
        {
            "shapes": [list(tensor.shape) for tensor in group],
            "dtypes": [str(tensor.dtype) for tensor in group],
            "finite": [
                (
                    bool(torch.isfinite(tensor).all())
                    if tensor.is_floating_point()
                    else True
                )
                for tensor in group
            ],
        }
        for group in metadata
    ]


def _graph_case(batch: int, group_name: str, rank: int, device: torch.device):
    buffers = list(_inputs(batch, "mixed", 0, device))

    def run():
        return _run_groups(*buffers[:4], group_name, rank, graph=True)[0]

    for _ in range(2):
        output = run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        output = run()
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    first = output.clone()

    updated = _inputs(batch, "mixed", 19, device)
    for target, source in zip(buffers, updated):
        target.copy_(source)
    graph.replay()
    torch.npu.synchronize()
    actual = output.clone()
    assert not torch.equal(first, actual)
    return _assert_oracle(actual, buffers[0], buffers[4])


def test_lite_official_moe_collective_ep8_board():
    if os.getenv("LITE_OFFICIAL_MOE_BOARD") != "1":
        import pytest

        pytest.skip("set LITE_OFFICIAL_MOE_BOARD=1 under an eight-rank torchrun")
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    if int(os.environ.get("WORLD_SIZE", "1")) != EP_SIZE:
        raise RuntimeError("official MoE collective board requires WORLD_SIZE=8")
    if int(os.getenv("HCCL_BUFFSIZE", "0")) < 128:
        raise RuntimeError("official MoE collective board requires HCCL_BUFFSIZE>=128")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)
    # The default group is control-only. Keep exactly one HCCL communicator,
    # dedicated to the official dispatch/combine pair.
    dist.init_process_group("gloo")
    moe_group = None
    artifact_dir = Path(os.getenv("LITE_OFFICIAL_MOE_ARTIFACT_DIR", "."))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        moe_group = dist.new_group(ranks=list(range(EP_SIZE)), backend="hccl")
        group_name = _hccl_name(moe_group, device)
        records = []
        for batch in (1, 2):
            for kind in ("real_only", "mixed", "identity_only", "hot"):
                inputs = _inputs(batch, kind, 0, device)
                output, metadata = _run_groups(
                    *inputs[:4], group_name, rank, graph=False
                )
                torch.npu.synchronize()
                records.append(
                    {
                        "mode": "eager",
                        "batch": batch,
                        "route": kind,
                        **_assert_oracle(output, inputs[0], inputs[4]),
                        "dispatch": _metadata_summary(metadata),
                    }
                )
        for batch in (1, 2):
            records.append(
                {
                    "mode": "graph",
                    "batch": batch,
                    "route": "mixed",
                    **_graph_case(batch, group_name, rank, device),
                }
            )
        payload = {
            "status": "passed",
            "rank": rank,
            "ep_size": EP_SIZE,
            "experts": EXPERTS,
            "local_experts": LOCAL_EXPERTS,
            "hidden": HIDDEN,
            "topk": TOPK,
            "records": records,
        }
        (artifact_dir / f"rank-{rank}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True)
        )
    except Exception as error:
        payload = {
            "status": "failed",
            "rank": rank,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        (artifact_dir / f"rank-{rank}-error.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True)
        )
        raise
    finally:
        if moe_group is not None:
            dist.destroy_process_group(moe_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    test_lite_official_moe_collective_ep8_board()
