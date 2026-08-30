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

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["group_moe_router", "group_moe_proj_output", "rmsnorm_scale"]


@triton.jit
def _group_moe_router_kernel(
    hidden_states_ptr,
    router_weight_ptr,
    output_ptr,
    num_tokens,
    num_experts_per_group,
    hidden_size_per_group,
    moe_group_size,
    stride_h_0,
    stride_h_1,
    stride_w_0,
    stride_w_1,
    stride_o_0,
    stride_o_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Per-group router GEMM kernel.

    Grid: ``(G, cdiv(T, BLOCK_M), cdiv(E_per_group, BLOCK_N))``. Each program
    computes one ``[BLOCK_M, BLOCK_N]`` output tile for one group, reading
    that group's slice of the hidden states and the matching column slice of
    the shared router weight.
    """
    group_id = tl.program_id(0)
    block_m_id = tl.program_id(1)
    block_n_id = tl.program_id(2)

    m_start = block_m_id * BLOCK_M
    n_start = block_n_id * BLOCK_N

    # Hidden rows for group ``group_id`` live at [group_id*T : (group_id+1)*T].
    h_m_start = group_id * num_tokens + m_start
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = (m_start + offs_m) < num_tokens
    n_mask = (n_start + offs_n) < num_experts_per_group

    # Router weight: [E_per_group, H]. Group g uses columns [g*H_g : (g+1)*H_g].
    # h_k_start iterates [0, H_g), and the weight column is g*H_g + h_k_start.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, hidden_size_per_group, BLOCK_K):
        k_mask = (k_start + offs_k) < hidden_size_per_group
        h_ptrs = hidden_states_ptr + (
            (h_m_start + offs_m[:, None]) * stride_h_0
            + (k_start + offs_k[None, :]) * stride_h_1
        )
        h_block = tl.load(h_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w_ptrs = router_weight_ptr + (
            (n_start + offs_n[None, :]) * stride_w_0
            + (group_id * hidden_size_per_group + k_start + offs_k[:, None])
            * stride_w_1
        )
        w_block = tl.load(w_ptrs, mask=n_mask[None, :] & k_mask[:, None], other=0.0)
        acc += tl.dot(h_block.to(tl.float32), w_block.to(tl.float32))

    out_ptrs = output_ptr + (
        (h_m_start + offs_m[:, None]) * stride_o_0
        + (n_start + offs_n[None, :]) * stride_o_1
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def group_moe_router(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    moe_group_size: int,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Multi-group router GEMM.

    Each group independently computes router logits using its own slice of the
    hidden states and a shared router weight matrix.

    Args:
        hidden_states: shape ``[T * G, H_g]``, RMSNorm-ed and already
            group-split along dim-0 (rows ``[g*T : (g+1)*T]`` are group ``g``).
            ``T`` is the token count, ``G = moe_group_size``,
            ``H_g = hidden_size // G``.
        router_weight: shape ``[E_per_group, H]``. Each group uses columns
            ``[g*H_g : (g+1)*H_g]`` as its own routing projection.
        moe_group_size: number of expert groups (``G``).
        out_dtype: output tensor dtype (typically ``torch.float32``).

    Returns:
        Router logits, shape ``[T * G, E_per_group]``, dtype ``out_dtype``.
        Rows ``[g*T : (g+1)*T]`` contain logits for group ``g``.
    """
    num_tokens = hidden_states.shape[0] // moe_group_size
    hidden_size_per_group = hidden_states.shape[1]
    num_experts_per_group = router_weight.shape[0]

    output = torch.empty(
        num_tokens * moe_group_size,
        num_experts_per_group,
        device=hidden_states.device,
        dtype=out_dtype,
    )

    block_m = 32
    block_n = 64
    block_k = 128

    grid = (
        moe_group_size,
        triton.cdiv(num_tokens, block_m),
        triton.cdiv(num_experts_per_group, block_n),
    )
    _group_moe_router_kernel[grid](
        hidden_states,
        router_weight,
        output,
        num_tokens,
        num_experts_per_group,
        hidden_size_per_group,
        moe_group_size,
        hidden_states.stride(0),
        hidden_states.stride(1),
        router_weight.stride(0),
        router_weight.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_stages=3,
        num_warps=4,
    )
    return output


@triton.jit
def _group_moe_proj_output_kernel(
    hidden_states_ptr,
    proj_weight_ptr,
    output_ptr,
    num_tokens,
    hidden_size_per_group,
    hidden_size,
    stride_h_0,
    stride_h_1,
    stride_w_0,
    stride_w_1,
    stride_o_0,
    stride_o_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Per-group output projection kernel.

    Grid: ``(cdiv(T, BLOCK_M), cdiv(H, BLOCK_N))``. For each output tile
    ``[BLOCK_M, BLOCK_N]`` it accumulates the contribution of every group:
    ``output[t, n] = sum_g(hidden[g*T+t, :] @ proj_weight[n, g*H_g:(g+1)*H_g].T)``.
    """
    block_m_id = tl.program_id(0)
    block_n_id = tl.program_id(1)

    m_start = block_m_id * BLOCK_M
    n_start = block_n_id * BLOCK_N

    m_end = tl.minimum(m_start + BLOCK_M, num_tokens)
    n_end = tl.minimum(n_start + BLOCK_N, hidden_size)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = (m_start + offs_m) < num_tokens
    n_mask = (n_start + offs_n) < hidden_size

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_block_k_per_group = hidden_size_per_group // BLOCK_K
    for w_k_start in range(0, hidden_size, BLOCK_K):
        block_k_id = w_k_start // BLOCK_K
        block_k_id_in_group = block_k_id % num_block_k_per_group
        group_id = block_k_id // num_block_k_per_group

        h_m_start = group_id * num_tokens + m_start
        h_k_start = block_k_id_in_group * BLOCK_K

        h_ptrs = hidden_states_ptr + (
            (h_m_start + offs_m[:, None]) * stride_h_0
            + (h_k_start + offs_k[None, :]) * stride_h_1
        )
        h_m_end = tl.minimum(m_start + BLOCK_M, num_tokens)
        h_k_end = tl.minimum(h_k_start + BLOCK_K, hidden_size_per_group)
        h_mask = ((m_start + offs_m)[:, None] < h_m_end) & (
            (h_k_start + offs_k)[None, :] < h_k_end
        )
        h_block = tl.load(h_ptrs, mask=h_mask, other=0.0)

        w_k_end = tl.minimum(w_k_start + BLOCK_K, hidden_size)
        w_ptrs = proj_weight_ptr + (
            (n_start + offs_n[None, :]) * stride_w_0
            + (w_k_start + offs_k[:, None]) * stride_w_1
        )
        w_mask = ((n_start + offs_n)[None, :] < n_end) & (
            (w_k_start + offs_k)[:, None] < w_k_end
        )
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(h_block.to(tl.float32), w_block.to(tl.float32))

    o_ptrs = output_ptr + (
        (m_start + offs_m[:, None]) * stride_o_0
        + (n_start + offs_n[None, :]) * stride_o_1
    )
    o_mask = ((m_start + offs_m)[:, None] < m_end) & (
        (n_start + offs_n)[None, :] < n_end
    )
    # fp32 accumulate, cast back to the output dtype for storage.
    tl.store(o_ptrs, acc.to(output_ptr.dtype.element_ty), mask=o_mask)


def group_moe_proj_output(
    hidden_states: torch.Tensor,
    proj_weight: torch.Tensor,
    moe_group_size: int,
) -> torch.Tensor:
    """Multi-group output projection.

    Gathers per-group MoE outputs and projects them back into the full hidden
    dimension via a shared weight matrix. This is the inverse of
    :func:`rmsnorm_scale`'s group-split output layout.

    Args:
        hidden_states: shape ``[T * G, H_g]``, per-group MoE outputs stacked
            along dim-0. Rows ``[g*T : (g+1)*T]`` are group ``g``'s output.
        proj_weight: shape ``[H, H]``, output projection weight
            (``nn.Linear`` weight, no bias).
        moe_group_size: number of expert groups (``G``).

    Returns:
        Projected hidden states, shape ``[T, H]``, dtype ``hidden_states.dtype``.
    """
    assert hidden_states.dim() == 2 and proj_weight.dim() == 2

    num_tokens = hidden_states.shape[0] // moe_group_size
    hidden_size_per_group = hidden_states.shape[1]
    hidden_size = hidden_size_per_group * moe_group_size

    output = torch.empty(
        num_tokens,
        hidden_size,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    block_m = 32
    block_n = 128
    block_k = 128

    assert hidden_size_per_group % block_n == 0

    grid = (triton.cdiv(num_tokens, block_m), triton.cdiv(hidden_size, block_n))
    _group_moe_proj_output_kernel[grid](
        hidden_states,
        proj_weight,
        output,
        num_tokens,
        hidden_size_per_group,
        hidden_size,
        hidden_states.stride(0),
        hidden_states.stride(1),
        proj_weight.stride(0),
        proj_weight.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_stages=3,
        num_warps=4,
    )
    return output


@triton.jit
def _rmsnorm_scale_kernel(
    hidden_states_ptr,
    norm_weight_ptr,
    output_ptr,
    hidden_states_stride_0,
    output_stride_0,
    num_tokens,
    hidden_size,
    hidden_size_per_group,
    eps,
    norm_scale,
    BLOCK_K: tl.constexpr,
):
    """Fused RMSNorm + scale + group-split kernel.

    One program per token: compute the RMSNorm variance over the full hidden
    dimension, normalize, multiply by the RMSNorm weight and ``norm_scale``,
    then scatter the result into the group-major output layout
    (rows ``[g*T + t]`` for group ``g``).
    """
    token_idx = tl.program_id(0)

    # --- variance over the full hidden dimension ---
    var_acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for offset in range(0, hidden_size, BLOCK_K):
        col_offsets = offset + tl.arange(0, BLOCK_K)
        mask = col_offsets < hidden_size
        x = tl.load(
            hidden_states_ptr + token_idx * hidden_states_stride_0 + col_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        var_acc += x * x

    variance = tl.sum(var_acc, axis=0) / hidden_size
    rrms = 1.0 / tl.sqrt(variance + eps)

    # --- normalize, scale, and scatter into the group-split output ---
    num_block_k_per_group = hidden_size_per_group // BLOCK_K
    for offset in range(0, hidden_size, BLOCK_K):
        col_offsets = offset + tl.arange(0, BLOCK_K)
        mask = col_offsets < hidden_size
        x = tl.load(
            hidden_states_ptr + token_idx * hidden_states_stride_0 + col_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(norm_weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        out = x * rrms * w * norm_scale

        block_k_id = offset // BLOCK_K
        group_id = block_k_id // num_block_k_per_group
        output_col = col_offsets - group_id * hidden_size_per_group
        output_token_offset = group_id * num_tokens * output_stride_0

        tl.store(
            output_ptr + output_token_offset + token_idx * output_stride_0 + output_col,
            out.to(output_ptr.dtype.element_ty),
            mask=mask,
        )


def rmsnorm_scale(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    moe_group_size: int,
    norm_scale: float,
) -> torch.Tensor:
    """Fused RMSNorm + scale + group-split.

    Performs RMSNorm over the full hidden dimension, multiplies by
    ``norm_scale``, and scatters the result into a ``[T * G, H // G]`` tensor
    where rows ``[g*T : (g+1)*T]`` hold the group-``g`` slice of the normalized
    hidden states. This is the group-split layout the per-group router and
    experts consume.

    Args:
        hidden_states: shape ``[T, H]``.
        norm_weight: shape ``[H]``, RMSNorm scale weight.
        eps: RMSNorm epsilon.
        moe_group_size: number of expert groups (``G``).
        norm_scale: scalar multiplier applied after normalization.

    Returns:
        Group-split normalized hidden states, shape ``[T * G, H // G]``,
        dtype ``hidden_states.dtype``.
    """
    assert hidden_states.dim() == 2
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    hidden_size_per_group = hidden_size // moe_group_size
    assert hidden_size == moe_group_size * hidden_size_per_group
    assert hidden_size_per_group % 128 == 0

    output = torch.empty(
        num_tokens * moe_group_size,
        hidden_size_per_group,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    block_k = 128
    grid = (num_tokens,)
    _rmsnorm_scale_kernel[grid](
        hidden_states,
        norm_weight,
        output,
        hidden_states.stride(0),
        output.stride(0),
        num_tokens,
        hidden_size,
        hidden_size_per_group,
        float(eps),
        float(norm_scale),
        BLOCK_K=block_k,
    )
    return output
