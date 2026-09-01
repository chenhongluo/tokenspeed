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

"""Weight-owning skeleton and strict checkpoint layout for Lite."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from tokenspeed.runtime.configs.lite_config import (
    LiteConfig,
    is_lite_replicated_mla_mapping,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.layers.moe.loader import build_moe_checkpoint_loader
from tokenspeed.runtime.layers.moe.schema import ExpertCheckpointSchema
from tokenspeed.runtime.layers.moe.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.weights.unquant import create_dense_weight_pair
from tokenspeed.runtime.layers.paged_attention import PagedAttention

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
_ROUTER_RE = re.compile(
    r"^mlp\.expert_groups\.(\d+)\.router\."
    r"(classifier\.weight|e_score_correction_bias)$"
)
_OE_RE = re.compile(r"^model\.ngram_embeddings\.(embedders|post_projs)\.(\d+)\.weight$")

_Parallel = Literal["dense", "linear"]


@dataclass(frozen=True)
class LiteWeightSpec:
    source_name: str
    target_name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    category: str
    parallel: _Parallel | None = None
    shard_axis: int | None = None
    expert_id: int | None = None
    component_id: int | None = None


class LiteCheckpointLayout:
    """Single source of truth for Lite source keys, shapes and placement."""

    def __init__(self, config: LiteConfig) -> None:
        self.config = config

    def iter_source_names(self) -> Iterator[str]:
        yield "model.embed_tokens.weight"
        yield "lm_head.weight"
        yield "model.norm.weight"
        for layer_id in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_id}"
            yield f"{prefix}.input_layernorm.weight"
            yield f"{prefix}.post_attention_layernorm.weight"
            for group_id in range(self.config.moe_group_size):
                router = f"{prefix}.mlp.expert_groups.{group_id}.router"
                yield f"{router}.classifier.weight"
                yield f"{router}.e_score_correction_bias"
            for expert_id in range(self.config.num_experts):
                expert = f"{prefix}.mlp.experts.{expert_id}"
                yield f"{expert}.gate_proj.weight"
                yield f"{expert}.up_proj.weight"
                yield f"{expert}.down_proj.weight"
            yield f"{prefix}.mlp.norm.weight"
            yield f"{prefix}.mlp.proj_input.weight"
            yield f"{prefix}.mlp.proj_output.weight"
            for projection in ("gate_proj", "up_proj", "down_proj"):
                yield f"{prefix}.mlp.shared_experts.{projection}.weight"

            attention = f"{prefix}.self_attn"
            if self.config.is_kda_layer(layer_id):
                core = f"{attention}.linear_core"
                for projection in ("q_proj", "k_proj", "v_proj", "g_proj.0"):
                    yield f"{core}.{projection}.weight"
                for projection in ("q_conv1d", "k_conv1d", "v_conv1d"):
                    yield f"{core}.{projection}.weight"
                for projection in ("b_proj", "f_proj"):
                    yield f"{core}.{projection}.0.weight"
                    yield f"{core}.{projection}.1.weight"
                yield f"{core}.A_log"
                yield f"{core}.dt_bias"
                yield f"{core}.o_norm.weight"
                yield f"{core}.o_proj.weight"
            else:
                for suffix in (
                    "g_proj.weight",
                    "q_a_proj.weight",
                    "q_a_layernorm.weight",
                    "q_b_proj.weight",
                    "kv_a_proj_with_mqa.weight",
                    "kv_a_layernorm.weight",
                    "kv_b_proj.weight",
                    "o_proj.weight",
                ):
                    yield f"{attention}.{suffix}"

        for table_id in range(self.config.oe_component_count):
            yield f"model.ngram_embeddings.embedders.{table_id}.weight"
            yield f"model.ngram_embeddings.post_projs.{table_id}.weight"

    def spec(self, name: str) -> LiteWeightSpec:
        config = self.config
        if name in {"model.embed_tokens.weight", "lm_head.weight"}:
            return LiteWeightSpec(
                name,
                name,
                (config.vocab_size, config.hidden_size),
                torch.bfloat16,
                "dense-shard",
                "dense",
                0,
            )
        if name == "model.norm.weight":
            return self._replicated(name, (config.hidden_size,))

        oe_match = _OE_RE.fullmatch(name)
        if oe_match:
            family, raw_id = oe_match.groups()
            table_id = int(raw_id)
            if table_id >= config.oe_component_count:
                raise ValueError(f"Unexpected Lite checkpoint weight {name!r}.")
            shape = (
                (config.oe_table_rows(table_id), config.oe_hidden_size)
                if family == "embedders"
                else (config.hidden_size, config.oe_hidden_size)
            )
            return LiteWeightSpec(
                name,
                (
                    name
                    if family == "embedders"
                    else "model.ngram_embeddings.projection"
                ),
                shape,
                torch.bfloat16,
                "host-oe" if family == "embedders" else "oe-projection",
                component_id=table_id,
            )

        layer_match = _LAYER_RE.fullmatch(name)
        if layer_match is None:
            raise ValueError(f"Unexpected Lite checkpoint weight {name!r}.")
        layer_id = int(layer_match.group(1))
        suffix = layer_match.group(2)
        if layer_id >= config.num_hidden_layers:
            raise ValueError(f"Unexpected Lite checkpoint weight {name!r}.")

        if suffix in {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.norm.weight",
        }:
            return self._replicated(name, (config.hidden_size,))

        router_match = _ROUTER_RE.fullmatch(suffix)
        if router_match:
            group_id, leaf = router_match.groups()
            if int(group_id) >= config.moe_group_size:
                raise ValueError(f"Unexpected Lite checkpoint weight {name!r}.")
            experts = config.n_routed_experts + config.zero_expert_num
            shape = (
                (experts, config.hidden_size // config.moe_group_size)
                if leaf == "classifier.weight"
                else (experts,)
            )
            return LiteWeightSpec(name, name, shape, torch.float32, "router")

        expert_match = _EXPERT_RE.fullmatch(suffix)
        if expert_match:
            expert_id = int(expert_match.group(1))
            projection = expert_match.group(2)
            if expert_id >= config.num_experts:
                raise ValueError(f"Unexpected Lite checkpoint weight {name!r}.")
            group_hidden = config.hidden_size // config.moe_group_size
            shape = (
                (group_hidden, config.expert_ffn_hidden_size)
                if projection == "down"
                else (config.expert_ffn_hidden_size, group_hidden)
            )
            packed_name = "w2_weight" if projection == "down" else "w13_weight"
            target = name.rsplit(".experts.", 1)[0] + f".experts.{packed_name}"
            return LiteWeightSpec(
                name,
                target,
                shape,
                torch.bfloat16,
                "expert-ep",
                expert_id=expert_id,
            )

        dense_shapes = {
            "mlp.proj_input.weight": (config.hidden_size, config.hidden_size),
            "mlp.proj_output.weight": (config.hidden_size, config.hidden_size),
            "mlp.shared_experts.gate_proj.weight": (
                config.ffn_hidden_size,
                config.hidden_size,
            ),
            "mlp.shared_experts.up_proj.weight": (
                config.ffn_hidden_size,
                config.hidden_size,
            ),
            "mlp.shared_experts.down_proj.weight": (
                config.hidden_size,
                config.ffn_hidden_size,
            ),
        }
        if suffix in dense_shapes:
            shard_axis = (
                1 if suffix.endswith(("proj_output.weight", "down_proj.weight")) else 0
            )
            return LiteWeightSpec(
                name,
                name,
                dense_shapes[suffix],
                torch.bfloat16,
                "dense-shard",
                "dense",
                shard_axis,
            )

        attention_prefix = "self_attn."
        if not suffix.startswith(attention_prefix):
            raise ValueError(f"Unexpected Lite checkpoint weight {name!r}.")
        attention_suffix = suffix[len(attention_prefix) :]
        if config.is_kda_layer(layer_id):
            return self._kda_spec(name, attention_suffix)
        return self._mla_spec(name, attention_suffix)

    def _kda_spec(self, name: str, suffix: str) -> LiteWeightSpec:
        config = self.config
        core = "linear_core."
        if not suffix.startswith(core):
            raise ValueError(f"Unexpected Lite KDA weight {name!r}.")
        leaf = suffix[len(core) :]
        projection = config.linear_num_heads * config.linear_head_dim
        shapes = {
            "q_proj.weight": (projection, config.hidden_size),
            "k_proj.weight": (projection, config.hidden_size),
            "v_proj.weight": (projection, config.hidden_size),
            "g_proj.0.weight": (projection, config.hidden_size),
            "q_conv1d.weight": (projection, 1, config.linear_conv_size),
            "k_conv1d.weight": (projection, 1, config.linear_conv_size),
            "v_conv1d.weight": (projection, 1, config.linear_conv_size),
            "b_proj.0.weight": (config.linear_head_dim, config.hidden_size),
            "b_proj.1.weight": (projection, config.linear_head_dim),
            "f_proj.0.weight": (config.linear_head_dim, config.hidden_size),
            "f_proj.1.weight": (projection, config.linear_head_dim),
            "A_log": (config.linear_num_heads,),
            "dt_bias": (projection,),
            "o_norm.weight": (config.linear_head_dim,),
            "o_proj.weight": (config.hidden_size, projection),
        }
        if leaf not in shapes:
            raise ValueError(f"Unexpected Lite KDA weight {name!r}.")
        packed_components = {
            "q_proj.weight": 0,
            "k_proj.weight": 1,
            "v_proj.weight": 2,
            "g_proj.0.weight": 3,
            "f_proj.0.weight": 4,
            "b_proj.0.weight": 5,
        }
        if leaf in packed_components:
            target = name.replace(f"{core}{leaf}", "input_projection.weight")
            replicated = leaf in {"f_proj.0.weight", "b_proj.0.weight"}
            return LiteWeightSpec(
                name,
                target,
                shapes[leaf],
                torch.bfloat16,
                "kda-packed-projection",
                None if replicated else "linear",
                None if replicated else 0,
                component_id=packed_components[leaf],
            )
        target_leaf = {
            "b_proj.1.weight": "beta_b_proj.weight",
            "f_proj.1.weight": "f_b_proj.weight",
        }.get(leaf, leaf)
        target = name.replace(f"{core}{leaf}", target_leaf)
        replicated = {"o_norm.weight"}
        shard_axis = 1 if leaf == "o_proj.weight" else 0
        return LiteWeightSpec(
            name,
            target,
            shapes[leaf],
            torch.float32 if leaf in {"A_log", "dt_bias"} else torch.bfloat16,
            "rename" if target != name else "linear-shard",
            None if leaf in replicated else "linear",
            None if leaf in replicated else shard_axis,
        )

    def _mla_spec(self, name: str, suffix: str) -> LiteWeightSpec:
        config = self.config
        qk_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        shapes = {
            "g_proj.weight": (
                config.num_attention_heads * config.v_head_dim,
                config.hidden_size,
            ),
            "q_a_proj.weight": (config.q_lora_rank, config.hidden_size),
            "q_a_layernorm.weight": (config.q_lora_rank,),
            "q_b_proj.weight": (
                config.num_attention_heads * qk_dim,
                config.q_lora_rank,
            ),
            "kv_a_proj_with_mqa.weight": (
                config.kv_lora_rank + config.qk_rope_head_dim,
                config.hidden_size,
            ),
            "kv_a_layernorm.weight": (config.kv_lora_rank,),
            "kv_b_proj.weight": (
                config.num_attention_heads
                * (config.qk_nope_head_dim + config.v_head_dim),
                config.kv_lora_rank,
            ),
            "o_proj.weight": (
                config.hidden_size,
                config.num_attention_heads * config.v_head_dim,
            ),
        }
        if suffix not in shapes:
            raise ValueError(f"Unexpected Lite MLA weight {name!r}.")
        return self._replicated(name, shapes[suffix])

    @staticmethod
    def _replicated(name: str, shape: tuple[int, ...]) -> LiteWeightSpec:
        return LiteWeightSpec(name, name, shape, torch.bfloat16, "replicated")


class _Weight(nn.Module):
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        weight_nz: Literal["standard", "transposed"] | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        self.weight_nz = weight_nz
        self._weight_nz_prepared = False
        self._weight_nz_transposed = False

    def process_weights_after_loading(self, _module: nn.Module | None = None) -> None:
        if (
            self.weight_nz is None
            or self._weight_nz_prepared
            or self.weight.device.type != "npu"
        ):
            return
        from tokenspeed.runtime.utils.env import global_server_args_dict

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


def _all_reduce(tensor: torch.Tensor, group: tuple[int, ...]) -> torch.Tensor:
    from tokenspeed.runtime.distributed.comm_ops import all_reduce

    return all_reduce(tensor, group)


def _logits_metadata(ctx: Any) -> Any:
    from tokenspeed.runtime.layers.logits_processor import LogitsMetadata

    return LogitsMetadata.from_forward_context(ctx)


class _Norm(_Weight):
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


class _LiteKDAMergedProjection(nn.Module):
    component_count = 6

    def __init__(self, hidden_size: int, local_projection: int, head_dim: int):
        super().__init__()
        self.local_projection = local_projection
        self.head_dim = head_dim
        self._rows = (
            local_projection,
            local_projection,
            local_projection,
            local_projection,
            head_dim,
            head_dim,
        )
        self._offsets = (
            0,
            local_projection,
            2 * local_projection,
            3 * local_projection,
            4 * local_projection,
            4 * local_projection + head_dim,
        )
        self.weight = nn.Parameter(
            torch.empty(sum(self._rows), hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )

    def load_component(self, component_id: int, local_weight: torch.Tensor) -> None:
        if component_id < 0 or component_id >= self.component_count:
            raise ValueError(f"Invalid Lite KDA projection component {component_id}.")
        rows = self._rows[component_id]
        expected = (rows, self.weight.shape[1])
        if (
            tuple(local_weight.shape) != expected
            or local_weight.dtype != self.weight.dtype
        ):
            raise ValueError(
                f"Lite KDA projection component {component_id} has "
                f"shape/dtype {tuple(local_weight.shape)}/{local_weight.dtype}; "
                f"expected {expected}/{self.weight.dtype}."
            )
        if not self.weight.is_meta:
            start = self._offsets[component_id]
            self.weight.data[start : start + rows].copy_(
                local_weight.to(device=self.weight.device)
            )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = F.linear(hidden_states, self.weight)
        projection = self.local_projection
        low_rank = self.head_dim
        return (
            projected[:, : 3 * projection].contiguous(),
            projected[:, 3 * projection : 4 * projection],
            projected[:, 4 * projection : 4 * projection + low_rank],
            projected[:, 4 * projection + low_rank :],
        )


class _VocabEmbedding(_Weight):
    def __init__(self, config: LiteConfig, mapping: Any) -> None:
        tp = mapping.dense.tp_size
        if config.vocab_size % tp:
            raise ValueError("Lite vocab size must be divisible by dense TP.")
        super().__init__((config.vocab_size // tp, config.hidden_size), torch.bfloat16)
        self.vocab_size = config.vocab_size
        self.tp_rank = mapping.dense.tp_rank
        self.tp_size = tp
        self.tp_group = mapping.dense.tp_group

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return F.embedding(input_ids.clamp(0, self.vocab_size - 1), self.weight)
        start = self.tp_rank * self.weight.shape[0]
        valid = (input_ids >= start) & (input_ids < start + self.weight.shape[0])
        local_ids = (input_ids - start).masked_fill(~valid, 0)
        output = F.embedding(local_ids, self.weight)
        output.masked_fill_(~valid.unsqueeze(-1), 0)
        return _all_reduce(output, self.tp_group)


class _Router(nn.Module):
    def __init__(self, hidden_size: int, experts: int) -> None:
        super().__init__()
        self.classifier = _Weight((experts, hidden_size), torch.float32)
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(experts, dtype=torch.float32), requires_grad=False
        )


class _ExpertGroup(nn.Module):
    def __init__(self, hidden_size: int, experts: int) -> None:
        super().__init__()
        self.router = _Router(hidden_size, experts)


class _SharedExpert(nn.Module):
    def __init__(self, config: LiteConfig, dense_tp_size: int) -> None:
        super().__init__()
        local_ffn = config.ffn_hidden_size // dense_tp_size
        self.gate_proj = _Weight((local_ffn, config.hidden_size), torch.bfloat16)
        self.up_proj = _Weight((local_ffn, config.hidden_size), torch.bfloat16)
        self.down_proj = _Weight(
            (config.hidden_size, local_ffn), torch.bfloat16, weight_nz="standard"
        )


class _PackedExperts(nn.Module):
    def __init__(self, config: LiteConfig, mapping: Any) -> None:
        super().__init__()
        group_hidden = config.hidden_size // config.moe_group_size
        spec = MoELayerSpec(
            top_k=config.moe_topk,
            num_experts=config.num_experts,
            num_local_experts=config.num_experts // mapping.moe.ep_size,
            hidden_size=group_hidden,
            intermediate_size=config.expert_ffn_hidden_size,
            activation="silu",
            tp_rank=0,
            tp_size=1,
            ep_rank=mapping.moe.ep_rank,
            ep_size=mapping.moe.ep_size,
        )
        self.num_experts = spec.num_experts
        self.num_local_experts = spec.num_local_experts
        self.hidden_size = spec.hidden_size
        self.intermediate_size = spec.intermediate_size
        self.ep_rank = spec.ep_rank
        self.ep_size = spec.ep_size
        create_dense_weight_pair(spec, self, params_dtype=torch.bfloat16)


class _LocalExpertView:
    def __init__(
        self, experts: _PackedExperts, start: int, stop: int, num_experts: int
    ) -> None:
        self.num_experts = num_experts
        self.num_local_experts = stop - start
        self.hidden_size = experts.hidden_size
        self.intermediate_size = experts.intermediate_size
        self.ep_rank = experts.ep_rank
        self.ep_size = experts.ep_size
        self.w13_weight = experts.w13_weight[start:stop]
        self.w2_weight = experts.w2_weight[start:stop]
        self._ascend_bf16_moe_weights_processed = True


class LiteGroupedMoEParameters(nn.Module):
    def __init__(self, config: LiteConfig, mapping: Any) -> None:
        super().__init__()
        dense_tp = mapping.dense.tp_size
        group_hidden = config.hidden_size // config.moe_group_size
        self.config = config
        self.mapping = mapping
        self.group_hidden = group_hidden
        experts_per_router = config.n_routed_experts + config.zero_expert_num
        self.expert_groups = nn.ModuleList(
            _ExpertGroup(group_hidden, experts_per_router)
            for _ in range(config.moe_group_size)
        )
        self.experts = _PackedExperts(config, mapping)
        self.norm = _Norm(config.hidden_size)
        self.proj_input = _Weight(
            (config.hidden_size // dense_tp, config.hidden_size), torch.bfloat16
        )
        self.proj_output = _Weight(
            (config.hidden_size, config.hidden_size // dense_tp),
            torch.bfloat16,
            weight_nz="standard",
        )
        self.shared_experts = _SharedExpert(config, dense_tp)
        self._moe_plan: dict[str, Any] | None = None
        self._expert_views: tuple[_LocalExpertView, ...] = ()

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
            _LocalExpertView(
                self.experts,
                group_id * local_per_group,
                (group_id + 1) * local_per_group,
                self.config.n_routed_experts,
            )
            for group_id in range(self.config.moe_group_size)
        )

    def _route(
        self, grouped: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_weights = []
        all_ids = []
        all_logits = []
        for group_id, expert_group in enumerate(self.expert_groups):
            router = expert_group.router
            logits = F.linear(
                grouped[:, group_id].float(), router.classifier.weight.float()
            )
            if logits.device.type == "npu":
                import tokenspeed_kernel

                weights, ids = tokenspeed_kernel.moe_softmax_bias_topk(
                    logits,
                    router.e_score_correction_bias,
                    self.config.moe_topk,
                    routed_scaling_factor=self.config.routed_scaling_factor,
                )
            else:
                probabilities = torch.softmax(logits, dim=-1)
                ids = torch.topk(
                    probabilities + router.e_score_correction_bias,
                    self.config.moe_topk,
                    dim=-1,
                    sorted=False,
                ).indices
                weights = probabilities.gather(1, ids)
                weights = weights * self.config.routed_scaling_factor
            all_weights.append(weights)
            all_ids.append(ids.to(torch.int32))
            all_logits.append(logits)
        return (
            torch.stack(all_weights),
            torch.stack(all_ids),
            torch.stack(all_logits),
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
                    "Lite Ascend Grouped MoE weights were not processed after loading."
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
                "Lite Prefill Grouped MoE requires rank-ordered global_sp_num_tokens."
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
        global_sp_num_tokens: list[int] | None = None,
    ) -> torch.Tensor:
        if self.mapping.world_size == 8:
            if self.mapping.attn.cp_size == 8:
                return self._prefill(hidden_states, global_sp_num_tokens)
            return self._decode(hidden_states)
        if hidden_states.shape[0] == 0:
            return hidden_states

        grouped = self._project_grouped(hidden_states)
        topk_weights, topk_ids, _ = self._route(grouped)
        routed = F.linear(
            self._reference_experts(grouped, topk_weights, topk_ids),
            self.proj_output.weight,
        )
        return routed + self._shared(hidden_states)


class LiteKDAParameters(nn.Module):
    def __init__(self, config: LiteConfig, mapping: Any, layer_id: int) -> None:
        super().__init__()
        tp = mapping.linear_attn.tp_size
        projection = config.linear_num_heads * config.linear_head_dim
        local_projection = projection // tp
        self.config = config
        self.mapping = mapping
        self.layer_id = layer_id
        self.local_num_heads = config.linear_num_heads // tp
        self.head_dim = config.linear_head_dim
        self.projection = projection
        self.input_projection = _LiteKDAMergedProjection(
            config.hidden_size, local_projection, config.linear_head_dim
        )
        self.q_conv1d = _Weight(
            (local_projection, 1, config.linear_conv_size), torch.bfloat16
        )
        self.k_conv1d = _Weight(
            (local_projection, 1, config.linear_conv_size), torch.bfloat16
        )
        self.v_conv1d = _Weight(
            (local_projection, 1, config.linear_conv_size), torch.bfloat16
        )
        self.conv_weights: torch.Tensor | None = None
        self.beta_b_proj = _Weight(
            (local_projection, config.linear_head_dim), torch.bfloat16
        )
        self.f_b_proj = _Weight(
            (local_projection, config.linear_head_dim), torch.bfloat16
        )
        self.A_log = nn.Parameter(
            torch.empty(config.linear_num_heads // tp, dtype=torch.float32),
            requires_grad=False,
        )
        self.dt_bias = nn.Parameter(
            torch.empty(local_projection, dtype=torch.float32), requires_grad=False
        )
        self.o_norm = _Norm(config.linear_head_dim)
        self.o_proj = _Weight(
            (config.hidden_size, local_projection),
            torch.bfloat16,
            weight_nz="standard",
        )

    def process_weights_after_loading(self, _module: nn.Module | None = None) -> None:
        if self.conv_weights is not None:
            return
        self.conv_weights = torch.cat(
            [
                conv.weight.squeeze(1)
                for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d)
            ],
            dim=0,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: Any,
        out_cache_loc: torch.Tensor,
        comm_manager: Any = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        del positions, comm_manager
        if hidden_states.shape[0] == 0:
            return hidden_states

        qkv, output_gate, f_a_out, beta_down = self.input_projection(hidden_states)
        beta_logits = F.linear(beta_down, self.beta_b_proj.weight)
        conv_weights = self.conv_weights
        if conv_weights is None:
            self.process_weights_after_loading()
            conv_weights = self.conv_weights
        assert conv_weights is not None

        core_output = ctx.attn_backend.forward(
            q=None,
            k=None,
            v=None,
            layer=None,
            out_cache_loc=out_cache_loc,
            token_to_kv_pool=ctx.token_to_kv_pool,
            forward_mode=ctx.forward_mode,
            bs=ctx.bs,
            mixed_qkv=qkv,
            conv_weights=conv_weights,
            bias=None,
            activation="silu",
            key_dim=self.projection,
            value_dim=self.projection,
            attention_tp_size=self.mapping.linear_attn.tp_size,
            head_k_dim=self.head_dim,
            head_v_dim=self.head_dim,
            f_a_out=f_a_out,
            f_b_weight=self.f_b_proj.weight,
            beta_raw=beta_logits,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            lower_bound=self.config.linear_attn_config["gate_lower_bound"],
            output_gate=None,
            norm_weight=None,
            norm_eps=None,
            layer_id=self.layer_id,
            seq_len=hidden_states.shape[0],
        ).reshape(-1, self.local_num_heads, self.head_dim)

        if core_output.device.type in ("cuda", "npu"):
            from tokenspeed_kernel.ops.activation.triton import (
                rmsnorm_gated_sigmoid,
            )

            gated = rmsnorm_gated_sigmoid(
                core_output.flatten(1).contiguous(),
                output_gate,
                self.o_norm.weight,
                self.config.rms_norm_eps,
                self.local_num_heads,
                self.head_dim,
            )
        else:
            core_fp32 = core_output.float()
            normalized = core_fp32 * torch.rsqrt(
                core_fp32.square().mean(dim=-1, keepdim=True) + self.config.rms_norm_eps
            )
            normalized = (normalized * self.o_norm.weight.float()).to(core_output.dtype)
            output_gate = output_gate.reshape_as(normalized)
            gated = normalized * torch.sigmoid(output_gate.float()).to(normalized.dtype)
            gated = gated.flatten(1)
        return F.linear(gated, self.o_proj.weight)


class LiteMLAParameters(nn.Module):
    def __init__(self, config: LiteConfig, mapping: Any, layer_id: int) -> None:
        super().__init__()
        qk_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.config = config
        self.mapping = mapping
        self.layer_id = layer_id
        self.num_heads = config.num_attention_heads
        self.qk_head_dim = qk_dim
        self.scaling = qk_dim**-0.5
        self.g_proj = _Weight(
            (config.num_attention_heads * config.v_head_dim, config.hidden_size),
            torch.bfloat16,
        )
        self.q_a_proj = _Weight(
            (config.q_lora_rank, config.hidden_size),
            torch.bfloat16,
            weight_nz="transposed",
        )
        self.q_a_layernorm = _Norm(config.q_lora_rank)
        self.q_b_proj = _Weight(
            (config.num_attention_heads * qk_dim, config.q_lora_rank),
            torch.bfloat16,
            weight_nz="standard",
        )
        self.kv_a_proj_with_mqa = _Weight(
            (config.kv_lora_rank + config.qk_rope_head_dim, config.hidden_size),
            torch.bfloat16,
            weight_nz="transposed",
        )
        self.kv_a_layernorm = _Norm(config.kv_lora_rank)
        self.kv_b_proj = _Weight(
            (
                config.num_attention_heads
                * (config.qk_nope_head_dim + config.v_head_dim),
                config.kv_lora_rank,
            ),
            torch.bfloat16,
        )
        self.o_proj = _Weight(
            (config.hidden_size, config.num_attention_heads * config.v_head_dim),
            torch.bfloat16,
            weight_nz="standard",
        )
        self.attn_mqa = PagedAttention(
            config.num_attention_heads,
            config.kv_lora_rank + config.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=config.kv_lora_rank,
            group_id=FULL_ATTENTION,
        )
        self.attn_mha = PagedAttention(
            config.num_attention_heads,
            qk_dim,
            self.scaling,
            num_kv_heads=config.num_attention_heads,
            layer_id=layer_id,
            v_head_dim=config.v_head_dim,
            group_id=FULL_ATTENTION,
        )
        self.register_buffer("w_kc", None, persistent=False)
        self.register_buffer("w_vc", None, persistent=False)

    def process_weights_after_loading(self, _module: nn.Module | None = None) -> None:
        if self.w_kc is not None:
            return
        if self.config.mla_scale_q_lora:
            self.q_a_layernorm.weight.data.mul_(
                math.sqrt(self.config.hidden_size / self.config.q_lora_rank)
            )
        if self.config.mla_scale_kv_lora:
            self.kv_a_layernorm.weight.data.mul_(
                math.sqrt(self.config.hidden_size / self.config.kv_lora_rank)
            )
        packed = self.kv_b_proj.weight.unflatten(
            0,
            (
                self.num_heads,
                self.config.qk_nope_head_dim + self.config.v_head_dim,
            ),
        )
        w_kc, w_vc = packed.split(
            [self.config.qk_nope_head_dim, self.config.v_head_dim], dim=1
        )
        self.w_kc = w_kc.contiguous()
        self.w_vc = w_vc.transpose(1, 2).contiguous()

    def _project_q_latent(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_a = self.q_a_proj(hidden_states)
        latent = self.kv_a_proj_with_mqa(hidden_states)
        kv_a = latent[..., : self.config.kv_lora_rank]
        if hidden_states.device.type == "npu":
            from tokenspeed_kernel.ops.attention import mla_normalize_project_query

            q = mla_normalize_project_query(
                q_a,
                kv_a,
                self.q_a_layernorm.weight,
                self.kv_a_layernorm.weight,
                self.q_b_proj.weight,
                eps=self.config.rms_norm_eps,
            ).query
        else:
            q_a_fp32 = q_a.float()
            q_norm = q_a_fp32 * torch.rsqrt(
                q_a_fp32.square().mean(dim=-1, keepdim=True) + self.config.rms_norm_eps
            )
            q = F.linear(
                (q_norm * self.q_a_layernorm.weight.float()).to(q_a.dtype),
                self.q_b_proj.weight,
            )
            kv_fp32 = kv_a.float()
            kv_a.copy_(
                (
                    kv_fp32
                    * torch.rsqrt(
                        kv_fp32.square().mean(dim=-1, keepdim=True)
                        + self.config.rms_norm_eps
                    )
                    * self.kv_a_layernorm.weight.float()
                ).to(kv_a.dtype)
            )
        return q, latent, F.linear(hidden_states, self.g_proj.weight)

    def _write_cache(
        self,
        latent: torch.Tensor,
        ctx: Any,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        out_cache_loc = ctx.attn_backend.select_out_cache_loc(
            self.attn_mqa, out_cache_loc, ctx.forward_mode
        )
        kv_a, auxiliary = latent.split(
            [self.config.kv_lora_rank, self.config.qk_rope_head_dim], dim=-1
        )
        ctx.token_to_kv_pool.set_mla_kv_buffer(
            self.attn_mqa,
            out_cache_loc,
            kv_a.unsqueeze(1),
            auxiliary.unsqueeze(1),
        )
        return out_cache_loc

    def _absorbed_attention(
        self,
        q: torch.Tensor,
        latent: torch.Tensor,
        gate: torch.Tensor,
        ctx: Any,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        assert self.w_kc is not None and self.w_vc is not None
        q = q.view(-1, self.num_heads, self.qk_head_dim)
        q_nope, q_auxiliary = q.split(
            [self.config.qk_nope_head_dim, self.config.qk_rope_head_dim], dim=-1
        )
        q_absorbed = torch.bmm(q_nope.transpose(0, 1), self.w_kc).transpose(0, 1)
        q_absorbed = torch.cat((q_absorbed, q_auxiliary), dim=-1)
        self._write_cache(latent, ctx, out_cache_loc)
        latent_output = self.attn_mqa(
            q_absorbed,
            None,
            None,
            ctx,
            out_cache_loc,
            save_kv_cache=False,
        ).view(-1, self.num_heads, self.config.kv_lora_rank)
        if latent_output.device.type == "npu":
            from tokenspeed_kernel.ops.attention import mla_project_value

            return mla_project_value(latent_output, self.w_vc, gate=gate)
        projected = torch.bmm(
            latent_output.transpose(0, 1).contiguous(), self.w_vc
        ).transpose(0, 1)
        projected = projected.reshape(latent_output.shape[0], -1)
        projected.mul_(torch.sigmoid(gate).to(projected.dtype))
        return projected

    def _explicit_prefill(
        self,
        q: torch.Tensor,
        latent: torch.Tensor,
        gate: torch.Tensor,
        ctx: Any,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        q = q.view(-1, self.num_heads, self.qk_head_dim)
        kv_a, auxiliary = latent.split(
            [self.config.kv_lora_rank, self.config.qk_rope_head_dim], dim=-1
        )
        kv = F.linear(kv_a, self.kv_b_proj.weight).view(
            -1,
            self.num_heads,
            self.config.qk_nope_head_dim + self.config.v_head_dim,
        )
        k_nope, v = kv.split(
            [self.config.qk_nope_head_dim, self.config.v_head_dim], dim=-1
        )
        k = torch.cat(
            (k_nope, auxiliary.unsqueeze(1).expand(-1, self.num_heads, -1)), dim=-1
        )
        self._write_cache(latent, ctx, out_cache_loc)
        output = self.attn_mha(
            q,
            k,
            v,
            ctx,
            out_cache_loc,
            save_kv_cache=False,
        ).view(q.shape[0], -1)
        output.mul_(torch.sigmoid(gate).to(output.dtype))
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: Any,
        out_cache_loc: torch.Tensor,
        comm_manager: Any = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        del positions, comm_manager
        if hidden_states.shape[0] == 0:
            return hidden_states
        if ctx.forward_mode.is_mixed() or ctx.forward_mode.is_idle():
            raise NotImplementedError(
                "Lite MLA phase 5 supports pure Prefill or Decode"
            )
        if self.w_kc is None:
            self.process_weights_after_loading()
        q, latent, gate = self._project_q_latent(hidden_states)
        cached_extend = ctx.forward_mode.is_extend() and getattr(
            ctx.attn_backend.chunked_prefill_metadata,
            "use_absorbed_cached_extend",
            False,
        )
        if ctx.forward_mode.is_decode() or cached_extend:
            output = self._absorbed_attention(q, latent, gate, ctx, out_cache_loc)
        else:
            output = self._explicit_prefill(q, latent, gate, ctx, out_cache_loc)
        return F.linear(output, self.o_proj.weight)


class LiteDecoderLayer(nn.Module):
    def __init__(self, config: LiteConfig, mapping: Any, layer_id: int) -> None:
        super().__init__()
        self.mapping = mapping
        self.is_kda = config.is_kda_layer(layer_id)
        self.input_layernorm = _Norm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = (
            LiteKDAParameters(config, mapping, layer_id)
            if self.is_kda
            else LiteMLAParameters(config, mapping, layer_id)
        )
        self.post_attention_layernorm = _Norm(config.hidden_size, config.rms_norm_eps)
        self.mlp = LiteGroupedMoEParameters(config, mapping)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: Any,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        if ctx.forward_mode.is_idle():
            return hidden_states

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
        )
        if self.is_kda and self.mapping.linear_attn.tp_size > 1:
            hidden_states = _all_reduce(
                hidden_states, self.mapping.linear_attn.tp_group
            )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        global_sp_num_tokens = (
            ctx.global_num_tokens
            if self.mapping.world_size == 8 and self.mapping.attn.cp_size == 8
            else None
        )
        hidden_states = self.mlp(
            hidden_states, global_sp_num_tokens=global_sp_num_tokens
        )
        return residual + hidden_states


class LiteNgramParameters(nn.Module):
    def __init__(self, config: LiteConfig) -> None:
        super().__init__()
        self.config = config
        self._special_token_ids = config.special_token_ids
        self.embedders = nn.ModuleList()
        for _ in range(config.oe_component_count):
            table = nn.Module()
            table.register_parameter(
                "weight",
                nn.Parameter(
                    torch.empty(0, device="cpu", dtype=torch.bfloat16),
                    requires_grad=False,
                ),
            )
            self.embedders.append(table)
        self.projection = nn.Parameter(
            torch.empty(
                config.oe_component_count,
                config.oe_hidden_size,
                config.hidden_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.register_buffer(
            "ignore_tokens",
            torch.tensor(config.special_token_ids, dtype=torch.int64),
            persistent=False,
        )
        self._runtime: LiteOEStatePreparer | None = None

    def _host_special_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        ignore = tokens.new_tensor(self._special_token_ids)
        return (tokens.unsqueeze(-1) == ignore).any(dim=-1)

    def ngram_ids(
        self,
        input_ids: torch.Tensor,
        initial_context: torch.Tensor,
        lengths: Iterable[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return local table IDs, special mask, and each request's raw tail."""
        if input_ids.device.type != "cpu" or initial_context.device.type != "cpu":
            raise ValueError("Lite OE host IDs and context must be CPU tensors.")
        if input_ids.ndim != 1 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Lite OE input_ids must be a flat CPU integer tensor.")
        if (
            initial_context.ndim != 2
            or initial_context.shape[1] != self.config.emb_neighbor_num - 1
            or initial_context.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Lite OE initial_context must have shape [batch, 3].")

        lengths_tensor = torch.as_tensor(tuple(lengths), dtype=torch.int64)
        if (
            lengths_tensor.ndim != 1
            or lengths_tensor.shape[0] != initial_context.shape[0]
            or bool((lengths_tensor < 0).any())
            or int(lengths_tensor.sum()) != input_ids.numel()
        ):
            raise ValueError("Lite OE ragged lengths must match the flat input.")
        tokens = input_ids.to(torch.int64)
        context = initial_context.to(torch.int64)
        if bool((tokens < 0).any()) or bool((tokens >= self.config.vocab_size).any()):
            raise ValueError(
                f"Lite OE input IDs must be inside [0,{self.config.vocab_size})."
            )
        if tokens.numel() == 0:
            return (
                torch.empty((0, self.config.oe_component_count), dtype=torch.int64),
                torch.empty(0, dtype=torch.bool),
                context.clone(),
            )

        starts = torch.cumsum(lengths_tensor, dim=0) - lengths_tensor
        request_ids = torch.repeat_interleave(
            torch.arange(len(lengths_tensor)), lengths_tensor
        )
        columns = torch.arange(tokens.numel()) - starts[request_ids]
        relative = torch.arange(-3, 1)
        virtual = columns.unsqueeze(1) + relative.unsqueeze(0)
        from_context = virtual < 0
        context_values = context[request_ids].gather(1, (virtual + 3).clamp(0, 2))
        token_indices = (starts[request_ids].unsqueeze(1) + virtual).clamp(
            0, tokens.numel() - 1
        )
        windows = torch.where(from_context, context_values, tokens[token_indices])
        special = self._host_special_mask(windows)
        clean = windows.masked_fill(special, 0)

        shifted = {
            distance: clean[:, 3 - distance]
            * (clean[:, 3 - distance :].ne(0).all(dim=1))
            for distance in range(1, self.config.emb_neighbor_num)
        }
        table_ids = []
        current = clean[:, -1]
        for order in range(2, self.config.emb_neighbor_num + 1):
            for split_id in range(self.config.emb_split_num):
                table_id = (order - 2) * self.config.emb_split_num + split_id
                rows = self.config.oe_table_rows(table_id)
                local_ids = current.clone()
                for distance in range(1, order):
                    local_ids.add_(
                        shifted[distance] * pow(self.config.vocab_size, distance, rows)
                    )
                table_ids.append(local_ids.remainder(rows))

        tail_virtual = lengths_tensor.unsqueeze(1) + torch.arange(-3, 0)
        tail_from_context = tail_virtual < 0
        tail_context = context.gather(1, (tail_virtual + 3).clamp(0, 2))
        tail_token_indices = (starts.unsqueeze(1) + tail_virtual).clamp(
            0, tokens.numel() - 1
        )
        final_context = torch.where(
            tail_from_context, tail_context, tokens[tail_token_indices]
        )
        return torch.stack(table_ids, dim=1), special[:, -1], final_context

    def lookup_host(self, local_ids: torch.Tensor) -> torch.Tensor:
        if (
            local_ids.device.type != "cpu"
            or local_ids.dtype != torch.int64
            or local_ids.ndim != 2
            or local_ids.shape[1] != self.config.oe_component_count
        ):
            raise ValueError("Lite OE lookup IDs must be CPU int64 [tokens, 12].")
        if local_ids.shape[0] == 0:
            return torch.empty(
                (0, self.config.oe_component_count, self.config.oe_hidden_size),
                dtype=torch.bfloat16,
            )
        outputs = []
        for table_id, table in enumerate(self.embedders):
            expected = (self.config.oe_table_rows(table_id), self.config.oe_hidden_size)
            if tuple(table.weight.shape) != expected:
                raise RuntimeError(
                    f"Lite OE table {table_id} is not loaded: "
                    f"got {tuple(table.weight.shape)}, expected {expected}."
                )
            ids = local_ids[:, table_id]
            if bool((ids < 0).any()) or bool((ids >= expected[0]).any()):
                raise ValueError(f"Lite OE table {table_id} received an invalid ID.")
            outputs.append(F.embedding(ids, table.weight))
        return torch.stack(outputs, dim=1)

    def project_and_merge(
        self,
        word_hidden_states: torch.Tensor,
        raw_oe: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        tokens = word_hidden_states.shape[0]
        expected_raw = (
            tokens,
            self.config.oe_component_count,
            self.config.oe_hidden_size,
        )
        if (
            word_hidden_states.ndim != 2
            or word_hidden_states.shape[1] != self.config.hidden_size
            or tuple(raw_oe.shape) != expected_raw
            or input_ids.ndim != 1
            or input_ids.shape[0] != tokens
        ):
            raise ValueError("Lite OE projection inputs have incompatible shapes.")
        if tokens == 0:
            return word_hidden_states
        projected = raw_oe.reshape(tokens, self.config.hidden_size) @ (
            self.projection.reshape(self.config.hidden_size, self.config.hidden_size)
        )
        merged = (word_hidden_states + projected) / math.sqrt(
            self.config.oe_component_count + 1
        )
        special = (input_ids.unsqueeze(-1) == self.ignore_tokens).any(dim=-1)
        return torch.where(special.unsqueeze(-1), word_hidden_states, merged)

    def initialize_runtime(
        self,
        *,
        context_pages: torch.Tensor,
        checkpoint_granularity: int,
        max_request_slots: int,
        max_graph_tokens: int,
        device: str,
    ) -> None:
        self._runtime = LiteOEStatePreparer(
            self,
            context_pages=context_pages,
            checkpoint_granularity=checkpoint_granularity,
            max_request_slots=max_request_slots,
            max_graph_tokens=max_graph_tokens,
            device=device,
        )

    def prepare_forward_op(
        self,
        forward_op: Any,
        *,
        resolved_input_ids: torch.Tensor,
        graph_tokens: int | None,
    ) -> torch.Tensor:
        if self._runtime is None:
            raise RuntimeError("Lite OE runtime is not initialized.")
        return self._runtime.prepare_forward_op(
            forward_op,
            resolved_input_ids=resolved_input_ids,
            graph_tokens=graph_tokens,
        )

    def prepared_raw_oe(self, num_tokens: int) -> torch.Tensor:
        if self._runtime is None:
            raise RuntimeError("Lite OE runtime is not initialized.")
        return self._runtime.prepared_raw_oe(num_tokens)


class LiteOEStatePreparer:
    """Slot-safe host OE lookup and fixed-address Decode staging."""

    def __init__(
        self,
        ngram: LiteNgramParameters,
        *,
        context_pages: torch.Tensor,
        checkpoint_granularity: int,
        max_request_slots: int,
        max_graph_tokens: int,
        device: str,
    ) -> None:
        if (
            context_pages.ndim != 2
            or context_pages.shape[1] != ngram.config.emb_neighbor_num - 1
            or context_pages.dtype != torch.int32
        ):
            raise ValueError("Lite OE context pages must be int32 [pages, 3].")
        if (
            checkpoint_granularity <= 0
            or max_request_slots <= 0
            or max_graph_tokens <= 0
        ):
            raise ValueError("Lite OE runtime sizes must be positive.")
        self.ngram = ngram
        self.context_pages = context_pages
        self.checkpoint_granularity = checkpoint_granularity
        self.max_request_slots = max_request_slots
        self.max_graph_tokens = max_graph_tokens
        self._owners: list[str | None] = [None] * max_request_slots
        self._lengths = [0] * max_request_slots
        self._contexts = torch.full(
            (max_request_slots, ngram.config.emb_neighbor_num - 1),
            ngram.config.eos_token_id,
            dtype=torch.int64,
        )
        self.staging = torch.zeros(
            (
                max_graph_tokens,
                ngram.config.oe_component_count,
                ngram.config.oe_hidden_size,
            ),
            dtype=torch.bfloat16,
            device=device,
        )
        self._prepared = self.staging
        self.restore_count = 0

    @staticmethod
    def _page_id(table: Any, row: int, slot: int) -> int:
        if slot < 0 or getattr(table, "ndim", None) != 2:
            raise ValueError("Lite OE block table must be two-dimensional.")
        if row >= table.shape[0] or slot >= table.shape[1]:
            raise ValueError("Lite OE block table does not cover the state slot.")
        page_id = int(table[row, slot])
        if page_id <= 0:
            raise ValueError("Lite OE active state must not use page 0 or padding.")
        return page_id

    def _initial_context(
        self,
        *,
        request_id: str,
        request_pool_index: int,
        before_length: int,
        table: Any,
        row: int,
    ) -> torch.Tensor:
        if not 0 <= request_pool_index < self.max_request_slots:
            raise ValueError("Lite OE request-pool index is out of range.")
        if before_length < 0:
            raise ValueError("Lite OE prefix length must be non-negative.")
        if (
            self._owners[request_pool_index] == request_id
            and self._lengths[request_pool_index] == before_length
        ):
            return self._contexts[request_pool_index]
        if before_length == 0:
            return torch.full(
                (self.ngram.config.emb_neighbor_num - 1,),
                self.ngram.config.eos_token_id,
                dtype=torch.int64,
            )
        page_id = self._page_id(
            table,
            row,
            (before_length - 1) // self.checkpoint_granularity,
        )
        if page_id >= self.context_pages.shape[0]:
            raise ValueError("Lite OE input state page is out of range.")
        self.restore_count += 1
        return self.context_pages[page_id].to(device="cpu", dtype=torch.int64)

    def prepare(
        self,
        *,
        request_ids: Iterable[str],
        request_pool_indices: Iterable[int],
        input_ids: torch.Tensor,
        lengths: Iterable[int],
        before_lengths: Iterable[int],
        block_table: Any,
        graph_tokens: int | None = None,
    ) -> torch.Tensor:
        request_ids = tuple(request_ids)
        request_pool_indices = tuple(request_pool_indices)
        lengths = tuple(lengths)
        before_lengths = tuple(before_lengths)
        batch = len(request_ids)
        if not (
            len(request_pool_indices)
            == len(lengths)
            == len(before_lengths)
            == batch
            == block_table.shape[0]
        ):
            raise ValueError("Lite OE request metadata lengths do not match.")
        if input_ids.device.type != "cpu":
            raise ValueError("Lite OE input IDs must remain on CPU for host lookup.")
        if sum(lengths) != input_ids.numel() or any(length < 0 for length in lengths):
            raise ValueError("Lite OE ragged lengths do not match input IDs.")

        contexts = torch.stack(
            [
                self._initial_context(
                    request_id=request_ids[row],
                    request_pool_index=request_pool_indices[row],
                    before_length=before_lengths[row],
                    table=block_table,
                    row=row,
                )
                for row in range(batch)
            ]
        )
        ids, _, final_contexts = self.ngram.ngram_ids(input_ids, contexts, lengths)

        output_pages: list[int | None] = []
        written_pages: dict[int, tuple[str, int]] = {}
        for row, length in enumerate(lengths):
            request_pool_index = request_pool_indices[row]
            request_id = request_ids[row]
            after_length = before_lengths[row] + length
            page_id = (
                self._page_id(
                    block_table,
                    row,
                    (after_length - 1) // self.checkpoint_granularity,
                )
                if length
                else None
            )
            if page_id is not None:
                owner = (request_id, request_pool_index)
                if page_id in written_pages and written_pages[page_id] != owner:
                    raise ValueError("Lite OE output state page has multiple owners.")
                if page_id >= self.context_pages.shape[0]:
                    raise ValueError("Lite OE output state page is out of range.")
                written_pages[page_id] = owner
            output_pages.append(page_id)

        for row, page_id in enumerate(output_pages):
            request_pool_index = request_pool_indices[row]
            request_id = request_ids[row]
            after_length = before_lengths[row] + lengths[row]
            if page_id is not None:
                self.context_pages[page_id].copy_(
                    final_contexts[row].to(
                        device=self.context_pages.device,
                        dtype=torch.int32,
                    ),
                    non_blocking=True,
                )
            self._owners[request_pool_index] = request_id
            self._lengths[request_pool_index] = after_length
            self._contexts[request_pool_index].copy_(final_contexts[row])

        raw_oe = self.ngram.lookup_host(ids)
        if graph_tokens is None:
            self._prepared = raw_oe.to(self.staging.device, non_blocking=True)
            return self._prepared
        if not input_ids.numel() <= graph_tokens <= self.max_graph_tokens:
            raise ValueError("Lite OE graph token count is outside the staging range.")
        self.staging[input_ids.numel() :].zero_()
        if input_ids.numel():
            self.staging[: input_ids.numel()].copy_(raw_oe, non_blocking=True)
        self._prepared = self.staging[:graph_tokens]
        return self._prepared

    def prepare_forward_op(
        self,
        forward_op: Any,
        *,
        resolved_input_ids: torch.Tensor,
        graph_tokens: int | None,
    ) -> torch.Tensor:
        request_ids = tuple(forward_op.request_ids)
        request_pool_indices = tuple(forward_op.request_pool_indices)
        lengths = tuple(forward_op.input_lengths)
        num_extends = forward_op.num_extends()
        decode_ids = tuple(forward_op.decode_input_ids)
        if len(decode_ids) != len(request_ids) - num_extends:
            raise ValueError("Lite OE decode IDs do not match the decode rows.")
        if any(not 0 <= slot < self.max_request_slots for slot in request_pool_indices):
            raise ValueError("Lite OE request-pool index is out of range.")
        if resolved_input_ids.ndim != 1 or resolved_input_ids.numel() != sum(lengths):
            raise ValueError("Lite OE resolved input IDs do not match ragged lengths.")
        if any(length != 1 for length in lengths[num_extends:]):
            raise RuntimeError("Lite OE speculative Decode is not supported yet.")
        tokens = resolved_input_ids.to(device="cpu", dtype=torch.int64)
        before_lengths = tuple(forward_op.extend_prefix_lens) + tuple(
            (
                self._lengths[slot]
                if self._owners[slot] == request_id
                else forward_op.prefill_lengths[row]
            )
            for row, (request_id, slot) in enumerate(
                zip(
                    request_ids[num_extends:],
                    request_pool_indices[num_extends:],
                    strict=True,
                ),
                start=num_extends,
            )
        )
        return self.prepare(
            request_ids=request_ids,
            request_pool_indices=request_pool_indices,
            input_ids=tokens,
            lengths=lengths,
            before_lengths=before_lengths,
            block_table=forward_op.block_tables_arrays()["lite_oe"],
            graph_tokens=graph_tokens,
        )

    def prepared_raw_oe(self, num_tokens: int) -> torch.Tensor:
        if not 0 <= num_tokens <= self._prepared.shape[0]:
            raise ValueError("Lite OE prepared tensor does not cover model input.")
        return self._prepared[:num_tokens]


class LiteModel(nn.Module):
    def __init__(self, config: LiteConfig, mapping: Any) -> None:
        super().__init__()
        self.embed_tokens = _VocabEmbedding(config, mapping)
        self.layers = nn.ModuleList(
            LiteDecoderLayer(config, mapping, layer_id)
            for layer_id in range(config.num_hidden_layers)
        )
        self.norm = _Norm(config.hidden_size, config.rms_norm_eps)
        self.ngram_embeddings = LiteNgramParameters(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: Any,
        out_cache_loc: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        hidden_states = self.embed_tokens(input_ids)
        raw_oe = self.ngram_embeddings.prepared_raw_oe(input_ids.numel())
        hidden_states = self.ngram_embeddings.project_and_merge(
            hidden_states,
            raw_oe,
            input_ids,
        )
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, ctx, out_cache_loc)
        return self.norm(hidden_states), None


class LiteForCausalLM(nn.Module):
    """Lite causal LM with strict checkpoint ownership."""

    def __init__(self, config: LiteConfig, mapping: Any, **_kwargs: Any) -> None:
        super().__init__()
        self._validate_mapping(mapping)
        self.config = config
        self.mapping = mapping
        self.layout = LiteCheckpointLayout(config)
        self.model = LiteModel(config, mapping)
        self.lm_head = _Weight(
            (config.vocab_size // mapping.dense.tp_size, config.hidden_size),
            torch.bfloat16,
        )
        self.logits_processor = self._create_logits_processor()

    def _create_logits_processor(self) -> Any | None:
        npu = getattr(torch, "npu", None)
        if not torch.cuda.is_available() and not (
            npu is not None and npu.is_available()
        ):
            return None
        from tokenspeed.runtime.layers.logits_processor import LogitsProcessor

        return LogitsProcessor(
            self.config,
            tp_rank=self.mapping.dense.tp_rank,
            tp_size=self.mapping.dense.tp_size,
            tp_group=self.mapping.dense.tp_group,
        )

    @staticmethod
    def _validate_mapping(mapping: Any) -> None:
        if mapping.world_size not in {1, 8}:
            raise ValueError(
                "Lite model construction supports only N=1 tests or N=8 roles."
            )
        expected = 1 if mapping.world_size == 1 else 8
        if mapping.world_size == 8:
            if is_lite_replicated_mla_mapping(mapping):
                dense_tp = 8
                attention_group = 8
            elif (mapping.attn.cp_size, mapping.attn.dp_size) == (8, 1):
                dense_tp = 1
                attention_group = 1
            elif (mapping.attn.cp_size, mapping.attn.dp_size) == (1, 8):
                dense_tp = 8
                attention_group = 1
            else:
                raise ValueError(
                    "Lite MLA requires CP8, attention-DP8, or the replicated "
                    "MLA TP8 topology within each role."
                )
        else:
            dense_tp = 1
            attention_group = 1
        actual = {
            "PP": mapping.pp_size,
            "dense TP": mapping.dense.tp_size,
            "KDA TP": mapping.linear_attn.tp_size,
            "MoE EP": mapping.moe.ep_size,
            "MLA execution group": mapping.attn.tp_size,
        }
        required = {
            "PP": 1,
            "dense TP": dense_tp,
            "KDA TP": expected,
            "MoE EP": expected,
            "MLA execution group": attention_group,
        }
        mismatches = [
            f"{name}={actual[name]} (expected {value})"
            for name, value in required.items()
            if actual[name] != value
        ]
        if mismatches:
            raise ValueError("Invalid Lite role mapping: " + ", ".join(mismatches))

    @torch.no_grad()
    def forward(
        self,
        ctx: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        **_kwargs: Any,
    ) -> Any:
        if self.logits_processor is None:
            raise RuntimeError(
                "Lite causal-LM forward requires an accelerator runtime."
            )
        hidden_states, _ = self.model(
            input_ids,
            positions,
            ctx,
            out_cache_loc,
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            _logits_metadata(ctx),
        )

    def initialize_external_inputs(
        self,
        *,
        token_to_kv_pool: Any,
        max_request_slots: int,
        max_graph_tokens: int,
        device: str,
    ) -> None:
        self.model.ngram_embeddings.initialize_runtime(
            context_pages=token_to_kv_pool.arena.field("layer.0.lite.oe.context"),
            checkpoint_granularity=token_to_kv_pool.arena.plan.prefix_granularity,
            max_request_slots=max_request_slots,
            max_graph_tokens=max_graph_tokens,
            device=device,
        )

    def prepare_external_inputs(
        self,
        forward_op: Any,
        *,
        resolved_input_ids: torch.Tensor,
        graph_tokens: int | None,
    ) -> torch.Tensor:
        return self.model.ngram_embeddings.prepare_forward_op(
            forward_op,
            resolved_input_ids=resolved_input_ids,
            graph_tokens=graph_tokens,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        expected_sources = set(self.layout.iter_source_names())
        params = dict(self.named_parameters())
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params,
            expert_schema=ExpertCheckpointSchema(),
            num_experts=self.config.num_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
            local_expert_ids=self._local_expert_ids(),
        )
        expected_targets = set(params)
        modules = dict(self.named_modules())
        seen_sources: set[str] = set()
        loaded_targets: set[str] = set()
        packed_components: dict[str, set[int]] = {}

        for name, loaded_weight in weights:
            if name in seen_sources:
                raise ValueError(f"Duplicate Lite checkpoint weight {name!r}.")
            seen_sources.add(name)
            spec = self.layout.spec(name)
            if name not in expected_sources:
                raise ValueError(f"Unexpected Lite checkpoint weight {name!r}.")
            if (
                tuple(loaded_weight.shape) != spec.shape
                or loaded_weight.dtype != spec.dtype
            ):
                raise ValueError(
                    f"Lite checkpoint weight {name!r} has "
                    f"shape/dtype {tuple(loaded_weight.shape)}/{loaded_weight.dtype}; "
                    f"expected {spec.shape}/{spec.dtype}."
                )
            if spec.expert_id is not None:
                if not moe_loader.is_expert_checkpoint_weight(name):
                    raise ValueError(f"Unexpected Lite expert weight {name!r}.")
                if moe_loader.matches(name):
                    loaded_targets.add(moe_loader.load(name, loaded_weight))
                continue
            if spec.category == "oe-projection":
                param = params.get(spec.target_name)
                if param is None or spec.component_id is None:
                    raise ValueError(
                        f"Missing Lite target parameter {spec.target_name!r}."
                    )
                expected = (
                    self.config.oe_component_count,
                    self.config.oe_hidden_size,
                    self.config.hidden_size,
                )
                if tuple(param.shape) != expected:
                    raise ValueError(
                        f"Lite target {spec.target_name!r} has shape "
                        f"{tuple(param.shape)}; expected {expected}."
                    )
                if not param.is_meta:
                    param[spec.component_id].copy_(
                        loaded_weight.t().to(device=param.device)
                    )
                loaded_targets.add(spec.target_name)
                continue
            if spec.category == "kda-packed-projection":
                if spec.component_id is None:
                    raise ValueError(
                        f"Missing Lite KDA component for {spec.source_name!r}."
                    )
                module_name = spec.target_name.removesuffix(".weight")
                module = modules.get(module_name)
                if not isinstance(module, _LiteKDAMergedProjection):
                    raise ValueError(
                        f"Missing Lite KDA projection target {module_name!r}."
                    )
                components = packed_components.setdefault(spec.target_name, set())
                if spec.component_id in components:
                    raise ValueError(
                        f"Duplicate Lite KDA projection component "
                        f"{spec.component_id} for {spec.target_name!r}."
                    )
                module.load_component(
                    spec.component_id, self._local_weight(spec, loaded_weight)
                )
                components.add(spec.component_id)
                if len(components) == module.component_count:
                    loaded_targets.add(spec.target_name)
                continue
            if spec.target_name in loaded_targets:
                raise ValueError(
                    f"Duplicate Lite target parameter {spec.target_name!r}."
                )
            param = params.get(spec.target_name)
            if param is None:
                raise ValueError(f"Missing Lite target parameter {spec.target_name!r}.")
            local_weight = self._local_weight(spec, loaded_weight)
            if spec.category == "host-oe":
                if local_weight.device.type != "cpu" or param.device.type != "cpu":
                    raise ValueError("Lite OE embedding tables must remain on CPU.")
                param.data = local_weight
                if (
                    param.untyped_storage().data_ptr()
                    != local_weight.untyped_storage().data_ptr()
                ):
                    raise RuntimeError("Lite OE host storage adoption did not alias.")
                loaded_targets.add(spec.target_name)
                continue
            if (
                tuple(param.shape) != tuple(local_weight.shape)
                or param.dtype != spec.dtype
            ):
                raise ValueError(
                    f"Lite target {spec.target_name!r} has "
                    f"shape/dtype {tuple(param.shape)}/{param.dtype}; expected "
                    f"{tuple(local_weight.shape)}/{spec.dtype}."
                )
            if not param.is_meta:
                param.copy_(local_weight.to(device=param.device))
            loaded_targets.add(spec.target_name)

        missing_sources = sorted(expected_sources.difference(seen_sources))
        if missing_sources:
            raise ValueError(
                f"Lite checkpoint is missing {len(missing_sources)} source weights; "
                f"first={missing_sources[0]!r}."
            )
        missing_targets = sorted(expected_targets.difference(loaded_targets))
        if missing_targets:
            raise ValueError(
                f"Lite rank is missing {len(missing_targets)} target parameters; "
                f"first={missing_targets[0]!r}."
            )
        return loaded_targets

    def _local_expert_ids(self) -> tuple[int, ...]:
        local_per_group = self.config.n_routed_experts // self.mapping.moe.ep_size
        start = self.mapping.moe.ep_rank * local_per_group
        return tuple(
            group_id * self.config.n_routed_experts + start + local_id
            for group_id in range(self.config.moe_group_size)
            for local_id in range(local_per_group)
        )

    def _local_weight(
        self, spec: LiteWeightSpec, loaded_weight: torch.Tensor
    ) -> torch.Tensor:
        if spec.parallel is None:
            return loaded_weight
        parallel = (
            self.mapping.dense if spec.parallel == "dense" else self.mapping.linear_attn
        )
        assert spec.shard_axis is not None
        shard_size = spec.shape[spec.shard_axis] // parallel.tp_size
        return loaded_weight.narrow(
            spec.shard_axis, parallel.tp_rank * shard_size, shard_size
        )


# Keep direct imports used by the checkpoint probes while the runtime registry
# uses the collision-free internal architecture name.
FLASHLocalForCausalLM = LiteForCausalLM
EntryClass = LiteForCausalLM
