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

from tokenspeed.runtime.configs.lite_config import (
    LiteConfig,
    is_lite_replicated_mla_mapping,
)
from tokenspeed.runtime.layers.moe.loader import build_moe_checkpoint_loader
from tokenspeed.runtime.layers.moe.schema import ExpertCheckpointSchema
from tokenspeed.runtime.layers.over_embedding import (
    CheckpointedTailOEStatePreparer,
    HostLongCatOverEmbedding,
)
from tokenspeed.runtime.models.flash_local_attention import (
    PackedFLASHLocalKDA,
    SeparateProjectionKimiLinearMLAAttention,
)
from tokenspeed.runtime.models.flash_local_moe import (
    PackedFLASHLocalMoE,
)
from tokenspeed.runtime.models.flash_local_moe import PackedRMSNorm as _Norm
from tokenspeed.runtime.models.flash_local_moe import PackedWeight as _Weight
from tokenspeed.runtime.models.flash_local_moe import (
    grouped_moe_local_expert_ids,
)

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
_ROUTER_RE = re.compile(
    r"^mlp\.expert_groups\.(\d+)\.router\."
    r"(classifier\.weight|e_score_correction_bias)$"
)
_OE_RE = re.compile(r"^model\.ngram_embeddings\.(embedders|post_projs)\.(\d+)\.weight$")

# Temporary import compatibility until Step 5 removes this model entry.
LiteNgramParameters = HostLongCatOverEmbedding
LiteOEStatePreparer = CheckpointedTailOEStatePreparer

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


def _all_reduce(tensor: torch.Tensor, group: tuple[int, ...]) -> torch.Tensor:
    from tokenspeed.runtime.distributed.comm_ops import all_reduce

    return all_reduce(tensor, group)


def _logits_metadata(ctx: Any) -> Any:
    from tokenspeed.runtime.layers.logits_processor import LogitsMetadata

    return LogitsMetadata.from_forward_context(ctx)


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


class LiteDecoderLayer(nn.Module):
    def __init__(self, config: LiteConfig, mapping: Any, layer_id: int) -> None:
        super().__init__()
        self.mapping = mapping
        self.is_kda = config.is_kda_layer(layer_id)
        self.input_layernorm = _Norm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = (
            PackedFLASHLocalKDA(config, mapping, layer_id)
            if self.is_kda
            else SeparateProjectionKimiLinearMLAAttention(
                config,
                mapping,
                layer_id=layer_id,
                prefix=f"model.layers.{layer_id}.self_attn",
            )
        )
        self.post_attention_layernorm = _Norm(config.hidden_size, config.rms_norm_eps)
        self.mlp = PackedFLASHLocalMoE(config, mapping)

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
        hidden_states = self.mlp(hidden_states, ctx=ctx)
        return residual + hidden_states


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

    supports_oe_table_placement = True
    oe_runtime_capabilities = {
        "npu": {"host": "cache-checkpointed-tail"},
    }

    def __init__(
        self,
        config: LiteConfig,
        mapping: Any,
        *,
        oe_table_placement: str | None = "host",
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        if oe_table_placement not in {None, "host"}:
            raise ValueError("The temporary Lite model only supports Host OE tables.")
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
                param = params.get(spec.target_name)
                weight_loader = getattr(param, "weight_loader", None)
                if param is None or weight_loader is None:
                    raise ValueError(
                        f"Missing Lite KDA projection target {spec.target_name!r}."
                    )
                components = packed_components.setdefault(spec.target_name, set())
                if spec.component_id in components:
                    raise ValueError(
                        f"Duplicate Lite KDA projection component "
                        f"{spec.component_id} for {spec.target_name!r}."
                    )
                weight_loader(
                    param,
                    self._local_weight(spec, loaded_weight),
                    spec.component_id,
                )
                components.add(spec.component_id)
                if len(components) == 6:
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
                with torch.no_grad():
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
        return grouped_moe_local_expert_ids(
            num_groups=self.config.moe_group_size,
            num_experts_per_group=self.config.n_routed_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
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
