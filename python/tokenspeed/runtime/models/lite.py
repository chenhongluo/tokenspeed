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

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from tokenspeed.runtime.configs.lite_config import LiteConfig

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
            return LiteWeightSpec(name, name, shape, torch.bfloat16, "host-oe")

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
            return LiteWeightSpec(
                name, name, shape, torch.bfloat16, "expert-ep", expert_id=expert_id
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
        target_leaf = {
            "g_proj.0.weight": "g_proj.weight",
            "f_proj.0.weight": "f_a_proj.weight",
            "f_proj.1.weight": "f_b_proj.weight",
        }.get(leaf, leaf)
        target = name.replace(f"{core}{leaf}", target_leaf)
        replicated = {"b_proj.0.weight", "f_proj.0.weight", "o_norm.weight"}
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
    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)


class _Norm(_Weight):
    def __init__(self, size: int) -> None:
        super().__init__((size,), torch.bfloat16)


class _Expert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = _Weight((intermediate_size, hidden_size), torch.bfloat16)
        self.up_proj = _Weight((intermediate_size, hidden_size), torch.bfloat16)
        self.down_proj = _Weight((hidden_size, intermediate_size), torch.bfloat16)


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
        self.down_proj = _Weight((config.hidden_size, local_ffn), torch.bfloat16)


class LiteGroupedMoEParameters(nn.Module):
    def __init__(self, config: LiteConfig, mapping: Any) -> None:
        super().__init__()
        dense_tp = mapping.dense.tp_size
        group_hidden = config.hidden_size // config.moe_group_size
        experts_per_router = config.n_routed_experts + config.zero_expert_num
        self.expert_groups = nn.ModuleList(
            _ExpertGroup(group_hidden, experts_per_router)
            for _ in range(config.moe_group_size)
        )
        local_experts = config.num_experts // mapping.moe.ep_size
        first_expert = mapping.moe.ep_rank * local_experts
        self.experts = nn.ModuleDict(
            {
                str(expert_id): _Expert(group_hidden, config.expert_ffn_hidden_size)
                for expert_id in range(first_expert, first_expert + local_experts)
            }
        )
        self.norm = _Norm(config.hidden_size)
        self.proj_input = _Weight(
            (config.hidden_size // dense_tp, config.hidden_size), torch.bfloat16
        )
        self.proj_output = _Weight(
            (config.hidden_size, config.hidden_size // dense_tp), torch.bfloat16
        )
        self.shared_experts = _SharedExpert(config, dense_tp)

    def forward(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise NotImplementedError(
            "Lite Grouped MoE execution is implemented in phase 6."
        )


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
        self.q_proj = _Weight((local_projection, config.hidden_size), torch.bfloat16)
        self.k_proj = _Weight((local_projection, config.hidden_size), torch.bfloat16)
        self.v_proj = _Weight((local_projection, config.hidden_size), torch.bfloat16)
        self.g_proj = _Weight((local_projection, config.hidden_size), torch.bfloat16)
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
        self.b_proj = nn.Sequential(
            _Weight((config.linear_head_dim, config.hidden_size), torch.bfloat16),
            _Weight((local_projection, config.linear_head_dim), torch.bfloat16),
        )
        self.f_a_proj = _Weight(
            (config.linear_head_dim, config.hidden_size), torch.bfloat16
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
        self.o_proj = _Weight((config.hidden_size, local_projection), torch.bfloat16)

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

        qkv = torch.cat(
            [
                F.linear(hidden_states, projection.weight)
                for projection in (self.q_proj, self.k_proj, self.v_proj)
            ],
            dim=-1,
        )
        output_gate = F.linear(hidden_states, self.g_proj.weight)
        f_a_out = F.linear(hidden_states, self.f_a_proj.weight)
        beta_down = F.linear(hidden_states, self.b_proj[0].weight)
        beta_logits = F.linear(beta_down, self.b_proj[1].weight)
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
    def __init__(self, config: LiteConfig) -> None:
        super().__init__()
        qk_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.g_proj = _Weight(
            (config.num_attention_heads * config.v_head_dim, config.hidden_size),
            torch.bfloat16,
        )
        self.q_a_proj = _Weight(
            (config.q_lora_rank, config.hidden_size), torch.bfloat16
        )
        self.q_a_layernorm = _Norm(config.q_lora_rank)
        self.q_b_proj = _Weight(
            (config.num_attention_heads * qk_dim, config.q_lora_rank), torch.bfloat16
        )
        self.kv_a_proj_with_mqa = _Weight(
            (config.kv_lora_rank + config.qk_rope_head_dim, config.hidden_size),
            torch.bfloat16,
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
        )

    def forward(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("Lite MLA execution is implemented in phase 5.")


class LiteDecoderLayer(nn.Module):
    def __init__(self, config: LiteConfig, mapping: Any, layer_id: int) -> None:
        super().__init__()
        self.input_layernorm = _Norm(config.hidden_size)
        self.self_attn = (
            LiteKDAParameters(config, mapping, layer_id)
            if config.is_kda_layer(layer_id)
            else LiteMLAParameters(config)
        )
        self.post_attention_layernorm = _Norm(config.hidden_size)
        self.mlp = LiteGroupedMoEParameters(config, mapping)

    def forward(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("Lite decoder execution is not connected yet.")


class LiteNgramParameters(nn.Module):
    def __init__(self, config: LiteConfig) -> None:
        super().__init__()
        self.embedders = nn.ModuleList()
        for _ in range(config.oe_component_count):
            table = nn.Module()
            table.register_parameter(
                "weight",
                nn.UninitializedParameter(
                    requires_grad=False, device="cpu", dtype=torch.bfloat16
                ),
            )
            self.embedders.append(table)
        self.post_projs = nn.ModuleList(
            _Weight((config.hidden_size, config.oe_hidden_size), torch.bfloat16)
            for _ in range(config.oe_component_count)
        )


class LiteModel(nn.Module):
    def __init__(self, config: LiteConfig, mapping: Any) -> None:
        super().__init__()
        dense_tp = mapping.dense.tp_size
        self.embed_tokens = _Weight(
            (config.vocab_size // dense_tp, config.hidden_size), torch.bfloat16
        )
        self.layers = nn.ModuleList(
            LiteDecoderLayer(config, mapping, layer_id)
            for layer_id in range(config.num_hidden_layers)
        )
        self.norm = _Norm(config.hidden_size)
        self.ngram_embeddings = LiteNgramParameters(config)


class FLASHLocalForCausalLM(nn.Module):
    """Phase-1 Lite model: exact parameter ownership and strict loading."""

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

    @staticmethod
    def _validate_mapping(mapping: Any) -> None:
        if mapping.world_size not in {1, 8}:
            raise ValueError(
                "Lite model construction supports only N=1 tests or N=8 roles."
            )
        expected = 1 if mapping.world_size == 1 else 8
        actual = {
            "PP": mapping.pp_size,
            "dense TP": mapping.dense.tp_size,
            "KDA TP": mapping.linear_attn.tp_size,
            "MoE EP": mapping.moe.ep_size,
            "MLA weight TP": mapping.attn.tp_size,
        }
        required = {
            "PP": 1,
            "dense TP": expected,
            "KDA TP": expected,
            "MoE EP": expected,
            "MLA weight TP": 1,
        }
        mismatches = [
            f"{name}={actual[name]} (expected {value})"
            for name, value in required.items()
            if actual[name] != value
        ]
        if mismatches:
            raise ValueError("Invalid Lite role mapping: " + ", ".join(mismatches))
        if mapping.world_size == 8 and mapping.attn.cp_size * mapping.attn.dp_size != 8:
            raise ValueError("Lite MLA requires CP8 or attention-DP8 within each role.")

    def forward(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("Lite numerical forward is connected in phases 3-7.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        expected_sources = set(self.layout.iter_source_names())
        params = dict(self.named_parameters())
        expected_targets = set(params)
        seen_sources: set[str] = set()
        loaded_targets: set[str] = set()

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
            if not self._owns(spec):
                continue
            if spec.target_name in loaded_targets:
                raise ValueError(
                    f"Duplicate Lite target parameter {spec.target_name!r}."
                )
            param = params.get(spec.target_name)
            if param is None:
                raise ValueError(f"Missing Lite target parameter {spec.target_name!r}.")
            local_weight = self._local_weight(spec, loaded_weight)
            if isinstance(param, nn.UninitializedParameter):
                param.materialize(local_weight.shape, device="cpu", dtype=spec.dtype)
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

    def _owns(self, spec: LiteWeightSpec) -> bool:
        if spec.expert_id is None:
            return True
        local_experts = self.config.num_experts // self.mapping.moe.ep_size
        start = self.mapping.moe.ep_rank * local_experts
        return start <= spec.expert_id < start + local_experts

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


EntryClass = FLASHLocalForCausalLM
