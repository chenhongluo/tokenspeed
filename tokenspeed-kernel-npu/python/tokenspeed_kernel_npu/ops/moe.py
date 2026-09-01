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

"""Ascend BF16 MoE primitives."""

from __future__ import annotations

import torch
import torch_npu


def ascend_softmax_bias_topk(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Lite's selection-only biased softmax TopK."""
    del enable_pdl
    weights, ids, _ = torch_npu.npu_moe_gating_top_k(
        router_logits,
        topk,
        bias=correction_bias,
        k_group=1,
        group_count=1,
        group_select_mode=0,
        renorm=0,
        norm_type=0,
        out_flag=False,
        routed_scaling_factor=routed_scaling_factor,
        eps=1e-20,
    )
    return weights.float(), ids.to(torch.int32)


def ascend_bf16_process_moe_weights(*, plan: dict, w: torch.nn.Module) -> None:
    """Convert canonical TokenSpeed weights to the Ascend GMM layout."""
    del plan
    if getattr(w, "_ascend_bf16_moe_weights_processed", False):
        return
    expected = (
        (w.num_local_experts, 2 * w.intermediate_size, w.hidden_size),
        (w.num_local_experts, w.hidden_size, w.intermediate_size),
    )
    if (tuple(w.w13_weight.shape), tuple(w.w2_weight.shape)) != expected:
        raise ValueError("Ascend MoE weights must use canonical W13/W2 layout")
    w.w13_weight.data = w.w13_weight.data.transpose(-1, -2).contiguous()
    w.w2_weight.data = w.w2_weight.data.transpose(-1, -2).contiguous()
    w._ascend_bf16_moe_weights_processed = True


def ascend_bf16_precomputed_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Apply the graph-stable Ascend local-expert chain."""
    del router_logits, num_tokens_global, max_num_tokens_per_gpu, enable_pdl
    if not do_finalize:
        raise ValueError("Ascend BF16 MoE does not support deferred finalization")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Ascend BF16 MoE requires precomputed top-k tensors")
    if x.ndim != 2 or topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("x and top-k tensors must be rank-2")
    if topk_ids.shape[0] != x.shape[0] or topk_ids.shape[1] == 0:
        raise ValueError("top-k tensors must have shape [tokens, top_k > 0]")
    if x.dtype != torch.bfloat16 or x.device.type != "npu":
        raise TypeError("Ascend BF16 MoE requires BF16 NPU hidden states")
    if topk_ids.device != x.device or topk_weights.device != x.device:
        raise ValueError("top-k tensors and hidden states must share a device")
    if plan.get("activation") not in {"silu", "swiglu"}:
        raise ValueError("Ascend BF16 MoE supports standard SwiGLU only")
    if not getattr(w, "_ascend_bf16_moe_weights_processed", False):
        raise RuntimeError("Ascend BF16 MoE weights were not processed after loading")
    if x.shape[0] == 0:
        return torch.empty_like(x)

    local_experts = int(w.num_local_experts)
    hidden_size = int(w.hidden_size)
    intermediate_size = int(w.intermediate_size)
    if tuple(w.w13_weight.shape) != (
        local_experts,
        hidden_size,
        2 * intermediate_size,
    ) or tuple(w.w2_weight.shape) != (
        local_experts,
        intermediate_size,
        hidden_size,
    ):
        raise ValueError("Ascend MoE weights have an incompatible GMM layout")
    if w.w13_weight.dtype != x.dtype or w.w2_weight.dtype != x.dtype:
        raise TypeError("hidden states and Ascend MoE weights must share a dtype")

    num_experts = int(w.num_experts)
    expert_lo = int(w.ep_rank) * local_experts
    expert_hi = expert_lo + local_experts
    if expert_hi > num_experts or int(w.ep_size) <= 1:
        raise ValueError("Ascend local-expert routing requires contiguous EP ownership")
    nonlocal_expert = expert_hi if expert_hi < num_experts else 0
    ids = topk_ids.to(torch.int32)
    ids = torch.where(
        (ids >= 0) & (ids < num_experts),
        ids,
        torch.full_like(ids, nonlocal_expert),
    )

    expanded, row_indices, expert_counts, _ = torch_npu.npu_moe_init_routing_v2(
        x,
        ids,
        scale=None,
        active_num=ids.numel(),
        expert_num=num_experts,
        quant_mode=-1,
        active_expert_range=[expert_lo, expert_hi],
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        row_idx_type=0,
        drop_pad_mode=0,
    )
    valid_rows = torch.arange(expanded.shape[0], device=x.device) < expert_counts.sum()
    expanded = expanded.masked_fill(~valid_rows.unsqueeze(-1), 0)
    counts = expert_counts.to(torch.int64)
    hidden = torch_npu.npu_grouped_matmul(
        [expanded],
        [w.w13_weight],
        bias=None,
        group_list=counts,
        split_item=3,
        output_dtype=torch.bfloat16,
        group_type=0,
        group_list_type=1,
    )[0].masked_fill(~valid_rows.unsqueeze(-1), 0)
    hidden = torch_npu.npu_swiglu(hidden, dim=-1).masked_fill(
        ~valid_rows.unsqueeze(-1), 0
    )
    hidden = torch_npu.npu_grouped_matmul(
        [hidden],
        [w.w2_weight],
        bias=None,
        group_list=counts,
        split_item=3,
        output_dtype=torch.bfloat16,
        group_type=0,
        group_list_type=1,
    )[0].masked_fill(~valid_rows.unsqueeze(-1), 0)
    return torch_npu.npu_moe_finalize_routing(
        hidden.unsqueeze(1),
        None,
        None,
        None,
        topk_weights.to(hidden.dtype),
        row_indices,
        ids,
        3,
    )


__all__ = [
    "ascend_bf16_precomputed_moe_apply",
    "ascend_bf16_process_moe_weights",
    "ascend_softmax_bias_topk",
]
