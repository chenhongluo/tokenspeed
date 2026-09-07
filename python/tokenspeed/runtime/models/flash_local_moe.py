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

"""Shared FLASHLocal Grouped MoE semantics and packed Ascend leaf."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal

import torch
from tokenspeed_kernel.platform import current_platform
from torch import nn
from torch.nn import functional as F

from tokenspeed.runtime.layers.moe.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.weights.unquant import create_dense_weight_pair
from tokenspeed.runtime.utils import set_weight_attrs
from tokenspeed.runtime.utils.env import global_server_args_dict


def flash_local_prefers_packed_moe() -> bool:
    """Select the Grouped MoE physical leaf at model construction.

    Ascend packs each rank's group-balanced EP-owned experts into ``w13/w2``
    containers for its fused executor. Other backends retain the existing
    ``MoELayer`` layout. Routing math and checkpoint source keys are shared.
    """
    return current_platform().is_npu


def grouped_moe_local_expert_ids(
    *,
    num_groups: int,
    num_experts_per_group: int,
    ep_rank: int,
    ep_size: int,
) -> tuple[int, ...]:
    """Return the group-balanced source expert IDs owned by one EP rank."""
    if (
        num_groups <= 0
        or num_experts_per_group <= 0
        or ep_size <= 0
        or num_experts_per_group % ep_size
        or not 0 <= ep_rank < ep_size
    ):
        raise ValueError("Grouped MoE experts must divide across valid EP ranks.")
    local_per_group = num_experts_per_group // ep_size
    start = ep_rank * local_per_group
    return tuple(
        group_id * num_experts_per_group + start + local_id
        for group_id in range(num_groups)
        for local_id in range(local_per_group)
    )


def grouped_moe_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply independent FP32 softmax/bias/top-k routing to every group.

    Args:
        router_logits: Group-major logits with shape ``[G, T, E]``.
        correction_bias: Selection-only bias with shape ``[G, E]`` or ``[G*E]``.
    """
    if router_logits.ndim != 3:
        raise ValueError("Grouped MoE router logits must have shape [G, T, E].")
    num_groups, _, num_experts = router_logits.shape
    if not 0 < top_k <= num_experts:
        raise ValueError("Grouped MoE top_k must be inside the router width.")
    if correction_bias.numel() != num_groups * num_experts:
        raise ValueError("Grouped MoE correction bias does not match router logits.")

    probabilities = torch.softmax(router_logits.float(), dim=-1)
    local_ids = torch.topk(
        probabilities + correction_bias.view(num_groups, 1, num_experts),
        top_k,
        dim=-1,
        sorted=False,
    ).indices
    weights = probabilities.gather(-1, local_ids)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * routed_scaling_factor, local_ids


class PackedWeight(nn.Module):
    """Unquantized parameter with optional Ascend Weight-NZ preparation."""

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        weight_nz: Literal["standard", "transposed"] | None = None,
        *,
        shard_axis: int | None = None,
        shard_rank: int = 0,
        shard_size: int = 1,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        self.weight_nz = weight_nz
        self.shard_axis = shard_axis
        self.shard_rank = shard_rank
        self.shard_size = shard_size
        self._weight_nz_prepared = False
        self._weight_nz_transposed = False
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if param is not self.weight:
            raise ValueError("Packed weight loader received an unrelated parameter.")
        if self.shard_axis is not None:
            if loaded_weight.shape[self.shard_axis] % self.shard_size:
                raise ValueError("Packed weight source does not divide across ranks.")
            shard_width = loaded_weight.shape[self.shard_axis] // self.shard_size
            loaded_weight = loaded_weight.narrow(
                self.shard_axis, self.shard_rank * shard_width, shard_width
            )
        if (
            tuple(param.shape) != tuple(loaded_weight.shape)
            or param.dtype != loaded_weight.dtype
        ):
            raise ValueError(
                "Packed weight shape/dtype mismatch: "
                f"target={tuple(param.shape)}/{param.dtype}, "
                f"source={tuple(loaded_weight.shape)}/{loaded_weight.dtype}."
            )
        if not param.is_meta:
            param.data.copy_(loaded_weight.to(device=param.device))

    def process_weights_after_loading(self, _module: nn.Module | None = None) -> None:
        if (
            self.weight_nz is None
            or self._weight_nz_prepared
            or self.weight.device.type != "npu"
        ):
            return
        if (
            not global_server_args_dict.get("npu_enable_weight_nz", False)
            or global_server_args_dict.get("disaggregation_mode") != "decode"
        ):
            return

        import tokenspeed_kernel

        transposed = self.weight_nz == "transposed"
        self.weight.data = tokenspeed_kernel.prepare_weight_nz(
            self.weight.data, transpose=transposed
        )
        self._weight_nz_transposed = transposed
        self._weight_nz_prepared = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._weight_nz_transposed:
            return torch.matmul(hidden_states, self.weight)
        return F.linear(hidden_states, self.weight)


class PackedRMSNorm(PackedWeight):
    def __init__(self, size: int, eps: float = 1e-6) -> None:
        super().__init__((size,), torch.bfloat16)
        self.variance_epsilon = eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape[0] == 0:
            if residual is not None:
                return hidden_states, residual
            return hidden_states
        if hidden_states.device.type in ("cuda", "npu"):
            from tokenspeed_kernel.ops.layernorm import rmsnorm

            return rmsnorm(
                hidden_states,
                self.weight.data,
                self.variance_epsilon,
                residual=residual,
            )
        normalized = hidden_states.float()
        if residual is not None:
            normalized = normalized + residual.float()
            residual.copy_(normalized.to(residual.dtype))
        output = (
            normalized
            * torch.rsqrt(
                normalized.square().mean(dim=-1, keepdim=True) + self.variance_epsilon
            )
            * self.weight.float()
        ).to(hidden_states.dtype)
        if residual is not None:
            return output, residual
        return output


class _PackedExpertGroup(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int) -> None:
        super().__init__()
        self.router = nn.Module()
        self.router.classifier = PackedWeight((num_experts, hidden_size), torch.float32)
        self.router.e_score_correction_bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )


class _PackedSharedExpert(nn.Module):
    def __init__(self, config: Any, mapping: Any) -> None:
        super().__init__()
        shared_count = getattr(
            config, "n_shared_experts", getattr(config, "num_shared_experts", 1)
        )
        shared_width = int(config.ffn_hidden_size) * int(shared_count)
        dense = mapping.dense
        local_ffn = shared_width // dense.tp_size
        self.gate_proj = PackedWeight(
            (local_ffn, config.hidden_size),
            torch.bfloat16,
            shard_axis=0,
            shard_rank=dense.tp_rank,
            shard_size=dense.tp_size,
        )
        self.up_proj = PackedWeight(
            (local_ffn, config.hidden_size),
            torch.bfloat16,
            shard_axis=0,
            shard_rank=dense.tp_rank,
            shard_size=dense.tp_size,
        )
        self.down_proj = PackedWeight(
            (config.hidden_size, local_ffn),
            torch.bfloat16,
            weight_nz="standard",
            shard_axis=1,
            shard_rank=dense.tp_rank,
            shard_size=dense.tp_size,
        )


class PackedFLASHLocalMoE(nn.Module):
    """FLASHLocal Grouped MoE with the validated packed Ascend layout.

    ``packed`` means EP-local runtime expert storage and the Ascend collective
    schedule; it does not mean a different checkpoint format or routing rule.
    """

    def __init__(self, config: Any, mapping: Any) -> None:
        super().__init__()
        dense_tp = mapping.dense.tp_size
        group_hidden = config.hidden_size // config.moe_group_size
        intermediate = config.expert_ffn_hidden_size
        total_experts = config.n_routed_experts * config.moe_group_size
        self.config = config
        self.mapping = mapping
        self.group_hidden = group_hidden
        self.hidden_size_per_group = group_hidden
        self.moe_group_size = config.moe_group_size
        self.n_routed_experts = config.n_routed_experts
        self.zero_expert_num = config.zero_expert_num
        self.local_expert_ids = grouped_moe_local_expert_ids(
            num_groups=config.moe_group_size,
            num_experts_per_group=config.n_routed_experts,
            ep_rank=mapping.moe.ep_rank,
            ep_size=mapping.moe.ep_size,
        )
        experts_per_router = config.n_routed_experts + config.zero_expert_num
        self.expert_groups = nn.ModuleList(
            _PackedExpertGroup(group_hidden, experts_per_router)
            for _ in range(config.moe_group_size)
        )

        self.experts = nn.Module()
        expert_spec = MoELayerSpec(
            top_k=config.moe_topk,
            num_experts=total_experts,
            num_local_experts=len(self.local_expert_ids),
            hidden_size=group_hidden,
            intermediate_size=intermediate,
            activation="silu",
            tp_rank=0,
            tp_size=1,
            ep_rank=mapping.moe.ep_rank,
            ep_size=mapping.moe.ep_size,
        )
        for field in (
            "num_experts",
            "num_local_experts",
            "hidden_size",
            "intermediate_size",
            "ep_rank",
            "ep_size",
        ):
            setattr(self.experts, field, getattr(expert_spec, field))
        create_dense_weight_pair(expert_spec, self.experts, params_dtype=torch.bfloat16)

        self.norm = PackedRMSNorm(config.hidden_size)
        self.proj_input = PackedWeight(
            (config.hidden_size // dense_tp, config.hidden_size),
            torch.bfloat16,
            shard_axis=0,
            shard_rank=mapping.dense.tp_rank,
            shard_size=dense_tp,
        )
        self.proj_output = PackedWeight(
            (config.hidden_size, config.hidden_size // dense_tp),
            torch.bfloat16,
            weight_nz="standard",
            shard_axis=1,
            shard_rank=mapping.dense.tp_rank,
            shard_size=dense_tp,
        )
        self.shared_experts = _PackedSharedExpert(config, mapping)
        self._moe_plan: dict[str, Any] | None = None
        self._expert_views: tuple[Any, ...] = ()

    def get_moe_routed_weights(self) -> list[torch.Tensor]:
        return [self.experts.w13_weight.data, self.experts.w2_weight.data]

    def load_group_router_weight(
        self, group_id: int, field: str, loaded_weight: torch.Tensor
    ) -> None:
        """Load one group router into its separate Ascend runtime module.

        Expert ownership is independent: every EP rank owns an equal slice from
        every group, as defined by ``grouped_moe_local_expert_ids``.
        """
        if not 0 <= group_id < len(self.expert_groups):
            raise ValueError(f"Invalid FLASHLocal router group {group_id}.")
        router = self.expert_groups[group_id].router
        target = (
            router.classifier.weight
            if field == "classifier.weight"
            else (
                router.e_score_correction_bias
                if field == "e_score_correction_bias"
                else None
            )
        )
        if target is None or tuple(target.shape) != tuple(loaded_weight.shape):
            raise ValueError(
                f"Invalid FLASHLocal router tensor {field!r} for group {group_id}."
            )
        target.data.copy_(loaded_weight.to(device=target.device))

    def process_weights_after_loading(self, _module: nn.Module | None = None) -> None:
        if self.experts.w13_weight.device.type != "npu" or self._expert_views:
            return

        import tokenspeed_kernel

        self._moe_plan = tokenspeed_kernel.moe_plan(
            "unquant",
            input_dtype=torch.bfloat16,
            activation="silu",
            routing_mode="precomputed_topk",
            ep_size=self.mapping.moe.ep_size,
            ispp=self.config.expert_ffn_hidden_size,
        )
        tokenspeed_kernel.moe_process_weights(self._moe_plan, self.experts)
        local_per_group = self.config.n_routed_experts // self.mapping.moe.ep_size
        self._expert_views = tuple(
            SimpleNamespace(
                num_experts=self.config.n_routed_experts,
                num_local_experts=local_per_group,
                hidden_size=self.experts.hidden_size,
                intermediate_size=self.experts.intermediate_size,
                ep_rank=self.experts.ep_rank,
                ep_size=self.experts.ep_size,
                w13_weight=self.experts.w13_weight[
                    group_id * local_per_group : (group_id + 1) * local_per_group
                ],
                w2_weight=self.experts.w2_weight[
                    group_id * local_per_group : (group_id + 1) * local_per_group
                ],
                _ascend_bf16_moe_weights_processed=True,
            )
            for group_id in range(self.config.moe_group_size)
        )

    def _route(
        self, grouped: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_logits = torch.stack(
            [
                F.linear(
                    grouped[:, group_id].float(),
                    expert_group.router.classifier.weight.float(),
                )
                for group_id, expert_group in enumerate(self.expert_groups)
            ]
        )
        if grouped.device.type != "npu":
            bias = torch.stack(
                [group.router.e_score_correction_bias for group in self.expert_groups]
            )
            weights, ids = grouped_moe_topk(
                all_logits,
                bias,
                top_k=self.config.moe_topk,
                renormalize=getattr(self.config, "norm_topk_prob", False),
                routed_scaling_factor=self.config.routed_scaling_factor,
            )
            return weights, ids.to(torch.int32), all_logits

        import tokenspeed_kernel

        routed = [
            tokenspeed_kernel.moe_softmax_bias_topk(
                all_logits[group_id],
                expert_group.router.e_score_correction_bias,
                self.config.moe_topk,
                routed_scaling_factor=self.config.routed_scaling_factor,
            )
            for group_id, expert_group in enumerate(self.expert_groups)
        ]
        if getattr(self.config, "norm_topk_prob", False):
            routed = [
                (
                    weights
                    / weights.sum(dim=-1, keepdim=True)
                    * self.config.routed_scaling_factor,
                    ids,
                )
                for weights, ids in routed
            ]
        return (
            torch.stack([result[0] for result in routed]),
            torch.stack([result[1].to(torch.int32) for result in routed]),
            all_logits,
        )

    def _reference_experts(
        self,
        grouped: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        group_outputs = []
        num_real = self.config.n_routed_experts
        intermediate = self.config.expert_ffn_hidden_size
        for group_id in range(self.config.moe_group_size):
            group_input = grouped[:, group_id]
            group_ids = topk_ids[group_id]
            group_weights = topk_weights[group_id]
            group_output = torch.zeros_like(group_input)
            identity_weight = torch.where(
                group_ids >= num_real,
                group_weights,
                torch.zeros_like(group_weights),
            ).sum(dim=-1, keepdim=True)
            group_output.add_(group_input * identity_weight.to(group_input.dtype))
            for expert_id_tensor in torch.unique(group_ids[group_ids < num_real]):
                expert_id = int(expert_id_tensor)
                token_ids, route_ids = torch.where(group_ids == expert_id)
                flat_expert_id = group_id * num_real + expert_id
                gate_up = F.linear(
                    group_input[token_ids],
                    self.experts.w13_weight[flat_expert_id],
                )
                gate, up = gate_up.split(intermediate, dim=-1)
                expert_output = F.linear(
                    F.silu(gate) * up,
                    self.experts.w2_weight[flat_expert_id],
                )
                expert_output = expert_output * group_weights[token_ids, route_ids].to(
                    expert_output.dtype
                ).unsqueeze(-1)
                group_output.index_add_(0, token_ids, expert_output)
            group_outputs.append(group_output)
        return torch.cat(group_outputs, dim=-1)

    def _project_grouped(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = F.linear(hidden_states, self.proj_input.weight)
        projected_fp32 = projected.float()
        projected = (
            projected_fp32
            * torch.rsqrt(
                projected_fp32.square().mean(dim=-1, keepdim=True)
                + self.config.rms_norm_eps
            )
            * self.norm.weight.float()
        ).to(projected.dtype)
        return (projected * self.config.grouped_moe_norm_scale).view(
            hidden_states.shape[0], self.config.moe_group_size, self.group_hidden
        )

    def _identity_routes(
        self,
        grouped: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        weights = torch.where(
            topk_ids >= self.config.n_routed_experts,
            topk_weights,
            torch.zeros_like(topk_weights),
        ).sum(dim=-1)
        return (
            grouped * weights.permute(1, 0).unsqueeze(-1).to(grouped.dtype)
        ).flatten(1)

    def _local_experts(
        self,
        grouped: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if grouped.device.type == "npu":
            if self._moe_plan is None or not self._expert_views:
                raise RuntimeError(
                    "FLASHLocal Ascend Grouped MoE weights were not processed."
                )
            import tokenspeed_kernel

            outputs = [
                tokenspeed_kernel.moe_apply(
                    self._moe_plan,
                    grouped[:, group_id],
                    self._expert_views[group_id],
                    topk_weights[group_id],
                    topk_weights=topk_weights[group_id],
                    topk_ids=topk_ids[group_id],
                )
                for group_id in range(self.config.moe_group_size)
            ]
            return torch.cat(outputs, dim=-1)

        outputs = []
        local_per_group = self.config.n_routed_experts // self.mapping.moe.ep_size
        expert_start = self.mapping.moe.ep_rank * local_per_group
        intermediate = self.config.expert_ffn_hidden_size
        for group_id in range(self.config.moe_group_size):
            group_input = grouped[:, group_id]
            group_output = torch.zeros_like(group_input)
            group_ids = topk_ids[group_id]
            group_weights = topk_weights[group_id]
            for expert_id_tensor in torch.unique(
                group_ids[
                    (group_ids >= expert_start)
                    & (group_ids < expert_start + local_per_group)
                ]
            ):
                expert_id = int(expert_id_tensor)
                token_ids, route_ids = torch.where(group_ids == expert_id)
                local_id = group_id * local_per_group + expert_id - expert_start
                gate_up = F.linear(
                    group_input[token_ids], self.experts.w13_weight[local_id]
                )
                gate, up = gate_up.split(intermediate, dim=-1)
                expert_output = F.linear(
                    F.silu(gate) * up, self.experts.w2_weight[local_id]
                )
                group_output.index_add_(
                    0,
                    token_ids,
                    expert_output
                    * group_weights[token_ids, route_ids]
                    .to(expert_output.dtype)
                    .unsqueeze(-1),
                )
            outputs.append(group_output)
        return torch.cat(outputs, dim=-1)

    def _shared(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shared_gate = F.linear(hidden_states, self.shared_experts.gate_proj.weight)
        shared_up = F.linear(hidden_states, self.shared_experts.up_proj.weight)
        return F.linear(
            F.silu(shared_gate) * shared_up,
            self.shared_experts.down_proj.weight,
        )

    def _prefill(
        self, hidden_states: torch.Tensor, global_sp_num_tokens: list[int] | None
    ) -> torch.Tensor:
        if (
            global_sp_num_tokens is None
            or len(global_sp_num_tokens) != self.mapping.moe.ep_size
            or any(tokens < 0 for tokens in global_sp_num_tokens)
            or global_sp_num_tokens[self.mapping.attn.cp_rank] != hidden_states.shape[0]
            or tuple(self.mapping.attn.cp_group) != tuple(self.mapping.moe.ep_group)
        ):
            raise ValueError(
                "FLASHLocal Prefill Grouped MoE requires rank-ordered "
                "global_num_tokens."
            )
        from tokenspeed.runtime.distributed.comm_ops import (
            token_all_gather,
            token_reduce_scatter,
        )

        grouped = self._project_grouped(hidden_states)
        topk_weights, topk_ids, _ = self._route(grouped)
        identity = self._identity_routes(grouped, topk_weights, topk_ids)
        group = self.mapping.moe.ep_group
        total_tokens = sum(global_sp_num_tokens)
        grouped = token_all_gather(
            grouped.flatten(1).contiguous(), group, global_sp_num_tokens
        ).view(total_tokens, self.config.moe_group_size, self.group_hidden)
        topk_weights = token_all_gather(
            topk_weights.permute(1, 0, 2).flatten(1).contiguous(),
            group,
            global_sp_num_tokens,
        ).view(total_tokens, self.config.moe_group_size, self.config.moe_topk)
        topk_weights = topk_weights.permute(1, 0, 2)
        topk_ids = token_all_gather(
            topk_ids.permute(1, 0, 2).flatten(1).contiguous(),
            group,
            global_sp_num_tokens,
        ).view(total_tokens, self.config.moe_group_size, self.config.moe_topk)
        topk_ids = topk_ids.permute(1, 0, 2)
        routed = token_reduce_scatter(
            self._local_experts(grouped, topk_weights, topk_ids),
            group,
            global_sp_num_tokens,
        )
        return F.linear(routed + identity, self.proj_output.weight) + self._shared(
            hidden_states
        )

    def _decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from tokenspeed.runtime.distributed.comm_ops import all_gather, all_reduce

        dense_group = self.mapping.dense.tp_group
        projected = all_gather(
            F.linear(hidden_states, self.proj_input.weight), dense_group, dim=-1
        )
        projected_fp32 = projected.float()
        projected = (
            projected_fp32
            * torch.rsqrt(
                projected_fp32.square().mean(dim=-1, keepdim=True)
                + self.config.rms_norm_eps
            )
            * self.norm.weight.float()
        ).to(projected.dtype)
        grouped = (projected * self.config.grouped_moe_norm_scale).view(
            hidden_states.shape[0], self.config.moe_group_size, self.group_hidden
        )
        topk_weights, topk_ids, _ = self._route(grouped)
        routed = all_reduce(
            self._local_experts(grouped, topk_weights, topk_ids),
            self.mapping.moe.ep_group,
        )
        routed = routed + self._identity_routes(grouped, topk_weights, topk_ids)
        local_hidden = self.config.hidden_size // self.mapping.dense.tp_size
        routed = routed.narrow(
            -1, self.mapping.dense.tp_rank * local_hidden, local_hidden
        )
        return all_reduce(
            F.linear(routed, self.proj_output.weight) + self._shared(hidden_states),
            dense_group,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_global_tokens: int = 0,
        max_num_tokens_per_gpu: int = 0,
        ctx: Any | None = None,
    ) -> torch.Tensor:
        del num_global_tokens, max_num_tokens_per_gpu
        if ctx is None:
            if hidden_states.shape[0] == 0:
                return hidden_states
            grouped = self._project_grouped(hidden_states)
            topk_weights, topk_ids, _ = self._route(grouped)
            routed = F.linear(
                self._reference_experts(grouped, topk_weights, topk_ids),
                self.proj_output.weight,
            )
            return routed + self._shared(hidden_states)

        mode = ctx.forward_mode
        if (
            mode is not None
            and mode.is_extend_or_mixed()
            and self.mapping.attn.cp_size > 1
        ):
            return self._prefill(hidden_states, ctx.global_num_tokens)
        if mode is not None and (
            mode.is_decode() or mode.is_idle() or mode.is_extend_or_mixed()
        ):
            return self._decode(hidden_states)
        raise ValueError("FLASHLocal Grouped MoE requires a valid ForwardContext mode.")


__all__ = [
    "PackedFLASHLocalMoE",
    "PackedRMSNorm",
    "PackedWeight",
    "flash_local_prefers_packed_moe",
    "grouped_moe_local_expert_ids",
    "grouped_moe_topk",
]
