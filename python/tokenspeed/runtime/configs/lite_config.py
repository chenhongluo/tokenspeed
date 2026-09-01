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

"""Configuration for the Lite hybrid KDA/MLA text model."""

from typing import Any

from transformers.configuration_utils import PretrainedConfig

from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)

_ARCHITECTURE = "FLASHLocalForCausalLM"
_ATTENTION_LAYER = "attention"

_REQUIRED_CHECKPOINT_FIELDS = frozenset(
    {
        "architectures",
        "attention_bias",
        "attention_dropout",
        "attention_method",
        "bos_token_id",
        "emb_neighbor_num",
        "emb_split_num",
        "eos_token_id",
        "expert_ffn_hidden_size",
        "fa_interval",
        "ffn_hidden_size",
        "grouped_moe_norm_scale",
        "hidden_size",
        "kda_nope",
        "kda_use_full_rank_gate",
        "kv_lora_rank",
        "linear_conv_size",
        "linear_head_dim",
        "linear_hidden_size",
        "linear_method",
        "linear_num_heads",
        "max_position_embeddings",
        "mla_scale_kv_lora",
        "mla_scale_q_lora",
        "mla_use_output_gate",
        "moe_group_size",
        "moe_impl",
        "moe_switch_token_num",
        "moe_topk",
        "n_routed_experts",
        "ngram_exclude_sp_token",
        "ngram_fix_normalize_factor",
        "ngram_vocab_size_ratio",
        "num_attention_heads",
        "num_layers",
        "q_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "rms_norm_eps",
        "routed_scaling_factor",
        "special_token_scope",
        "use_cache",
        "use_mla",
        "v_head_dim",
        "vocab_size",
        "zero_expert_num",
        "zero_expert_type",
    }
)


def is_lite_replicated_mla_mapping(mapping: Any) -> bool:
    """Whether an eight-rank Lite world replicates MLA for bounded serving."""
    return (
        mapping.world_size == 8
        and mapping.pp_size == 1
        and (
            mapping.attn.tp_size,
            mapping.attn.cp_size,
            mapping.attn.dp_size,
        )
        == (8, 1, 1)
        and mapping.dense.tp_size == 8
        and mapping.linear_attn.tp_size == 8
        and (mapping.moe.tp_size, mapping.moe.ep_size) == (1, 8)
    )


def lite_mla_component_tp_size(mapping: Any) -> int:
    """Return the MLA head-sharding width for a Lite execution mapping."""
    return 1 if is_lite_replicated_mla_mapping(mapping) else mapping.attn.tp_size


class LiteConfig(PretrainedConfig):
    """Native config for ``FLASHLocalForCausalLM`` checkpoints.

    The released checkpoint has no ``model_type``. ``get_config`` selects this
    class by architecture and ``from_dict`` validates the raw checkpoint before
    constructor defaults can hide a missing math field.
    """

    model_type = "lite"
    runtime_architecture = "LiteForCausalLM"

    @classmethod
    def matches_checkpoint(cls, config_dict: dict[str, Any]) -> bool:
        return _REQUIRED_CHECKPOINT_FIELDS.issubset(config_dict)

    def __init__(
        self,
        vocab_size: int = 163840,
        hidden_size: int = 3072,
        ffn_hidden_size: int = 3072,
        expert_ffn_hidden_size: int = 512,
        num_layers: int = 28,
        num_attention_heads: int = 32,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 1536,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        qk_nope_head_dim: int = 128,
        mla_scale_q_lora: bool = True,
        mla_scale_kv_lora: bool = True,
        routed_scaling_factor: float = 6.0,
        n_routed_experts: int = 384,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        zero_expert_num: int = 32,
        zero_expert_type: str = "identity",
        moe_topk: int = 12,
        moe_switch_token_num: int = 1024,
        moe_impl: str = "mix",
        ngram_vocab_size_ratio: float = 28.8,
        emb_neighbor_num: int = 4,
        emb_split_num: int = 4,
        ngram_exclude_sp_token: bool = True,
        special_token_scope: str = "0:4,36:55",
        ngram_fix_normalize_factor: bool = True,
        moe_group_size: int = 4,
        grouped_moe_norm_scale: float = 2.0,
        use_mla: int = 1,
        attention_method: str = "MLA",
        fa_interval: int = 4,
        linear_method: str = "FGBKDA",
        kda_nope: bool = True,
        linear_hidden_size: int = 3072,
        linear_head_dim: int = 128,
        linear_num_heads: int = 32,
        linear_conv_size: int = 4,
        mla_use_output_gate: bool = True,
        kda_use_full_rank_gate: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.intermediate_size = ffn_hidden_size
        self.expert_ffn_hidden_size = expert_ffn_hidden_size
        self.moe_intermediate_size = expert_ffn_hidden_size
        self.shared_expert_intermediate_size = ffn_hidden_size
        self.num_layers = num_layers
        self.num_hidden_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads

        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.mla_scale_q_lora = mla_scale_q_lora
        self.mla_scale_kv_lora = mla_scale_kv_lora
        self.mla_use_nope = kda_nope
        self.mla_use_output_gate = mla_use_output_gate

        self.routed_scaling_factor = routed_scaling_factor
        self.n_routed_experts = n_routed_experts
        self.num_experts = n_routed_experts * moe_group_size
        self.num_experts_per_tok = moe_topk
        self.num_shared_experts = 1
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type
        self.moe_topk = moe_topk
        self.moe_switch_token_num = moe_switch_token_num
        self.moe_impl = moe_impl
        self.moe_group_size = moe_group_size
        self.grouped_moe_norm_scale = grouped_moe_norm_scale

        self.ngram_vocab_size_ratio = ngram_vocab_size_ratio
        self.emb_neighbor_num = emb_neighbor_num
        self.emb_split_num = emb_split_num
        self.ngram_exclude_sp_token = ngram_exclude_sp_token
        self.special_token_scope = special_token_scope
        self.ngram_fix_normalize_factor = ngram_fix_normalize_factor

        self.use_mla = use_mla
        self.attention_method = attention_method
        self.fa_interval = fa_interval
        self.linear_method = linear_method
        self.kda_nope = kda_nope
        self.linear_hidden_size = linear_hidden_size
        self.linear_head_dim = linear_head_dim
        self.linear_num_heads = linear_num_heads
        self.linear_conv_size = linear_conv_size
        self.kda_use_full_rank_gate = kda_use_full_rank_gate
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_act = "silu"
        self.router_bias = False
        self.router_dtype = "float32"
        self.norm_topk_prob = False

        self._validate()
        self.linear_attn_config = {
            "kda_layers": [i + 1 for i in self.linear_layer_ids],
            "num_heads": linear_num_heads,
            "head_dim": linear_head_dim,
            "short_conv_kernel_size": linear_conv_size,
            "use_full_rank_gate": kda_use_full_rank_gate,
            "gate_lower_bound": -5.0,
        }
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=False,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any) -> "LiteConfig":
        missing = sorted(_REQUIRED_CHECKPOINT_FIELDS.difference(config_dict))
        if missing:
            raise ValueError(
                "Lite checkpoint config is missing required fields: "
                + ", ".join(missing)
            )
        if config_dict["architectures"] != [_ARCHITECTURE]:
            raise ValueError(f"Lite architectures must be [{_ARCHITECTURE!r}].")
        return super().from_dict(config_dict, **kwargs)

    def _validate(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "ffn_hidden_size": self.ffn_hidden_size,
            "expert_ffn_hidden_size": self.expert_ffn_hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "linear_head_dim": self.linear_head_dim,
            "linear_num_heads": self.linear_num_heads,
            "linear_conv_size": self.linear_conv_size,
            "fa_interval": self.fa_interval,
            "moe_group_size": self.moe_group_size,
            "n_routed_experts": self.n_routed_experts,
            "zero_expert_num": self.zero_expert_num,
            "moe_topk": self.moe_topk,
            "moe_switch_token_num": self.moe_switch_token_num,
            "emb_neighbor_num": self.emb_neighbor_num,
            "emb_split_num": self.emb_split_num,
            "ngram_vocab_size_ratio": self.ngram_vocab_size_ratio,
            "grouped_moe_norm_scale": self.grouped_moe_norm_scale,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(
                "Lite config fields must be positive: " + ", ".join(invalid)
            )
        if self.num_layers % self.fa_interval:
            raise ValueError("num_layers must be divisible by fa_interval.")
        if self.linear_hidden_size != self.hidden_size:
            raise ValueError("linear_hidden_size must equal hidden_size.")
        if self.linear_num_heads % 8:
            raise ValueError("linear_num_heads must be divisible by KDA TP8.")
        if self.num_experts % 8:
            raise ValueError("Grouped MoE real experts must be divisible by EP8.")
        if self.hidden_size % self.moe_group_size:
            raise ValueError("hidden_size must be divisible by moe_group_size.")
        if self.oe_component_count <= 0 or self.hidden_size % self.oe_component_count:
            raise ValueError("hidden_size must be divisible by the OE component count.")
        if self.moe_topk > self.n_routed_experts + self.zero_expert_num:
            raise ValueError("moe_topk exceeds the experts available in one group.")
        if self.linear_method.upper() != "FGBKDA" or not self.kda_use_full_rank_gate:
            raise ValueError("Lite requires FGBKDA with the full-rank output gate.")
        if not (self.kda_nope and self.mla_use_output_gate):
            raise ValueError("Lite requires NoPE MLA with the output gate enabled.")
        if self.use_mla != 1 or self.attention_method.upper() != "MLA":
            raise ValueError("Lite requires the hybrid MLA attention path.")
        if self.zero_expert_type != "identity":
            raise ValueError("Lite zero experts must use identity semantics.")
        if self.moe_impl != "mix" or not self.use_cache:
            raise ValueError("Lite requires mix MoE and cache support.")
        if self.attention_bias or self.attention_dropout != 0:
            raise ValueError("Lite attention must be bias-free with zero dropout.")
        if not (
            self.mla_scale_q_lora
            and self.mla_scale_kv_lora
            and self.ngram_exclude_sp_token
            and self.ngram_fix_normalize_factor
        ):
            raise ValueError("Lite checkpoint scaling and OE switches must be enabled.")
        self.special_token_ids

    def is_kda_layer(self, layer_idx: int) -> bool:
        if not 0 <= layer_idx < self.num_hidden_layers:
            raise IndexError(f"Layer index {layer_idx} is outside the model.")
        return (layer_idx + 1) % self.fa_interval != 0

    @property
    def layers_block_type(self) -> list[str]:
        return [
            LINEAR_ATTENTION if self.is_kda_layer(i) else _ATTENTION_LAYER
            for i in range(self.num_hidden_layers)
        ]

    @property
    def layer_types(self) -> list[str]:
        return [
            FULL_ATTENTION if layer_type == _ATTENTION_LAYER else layer_type
            for layer_type in self.layers_block_type
        ]

    @property
    def linear_layer_ids(self) -> list[int]:
        return [i for i in range(self.num_hidden_layers) if self.is_kda_layer(i)]

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [i for i in range(self.num_hidden_layers) if not self.is_kda_layer(i)]

    @property
    def oe_component_count(self) -> int:
        return (self.emb_neighbor_num - 1) * self.emb_split_num

    @property
    def oe_hidden_size(self) -> int:
        return self.hidden_size // self.oe_component_count

    @property
    def oe_table_base_rows(self) -> int:
        return int(self.vocab_size * self.ngram_vocab_size_ratio)

    def oe_table_rows(self, table_id: int) -> int:
        if not 0 <= table_id < self.oe_component_count:
            raise IndexError(f"OE table index {table_id} is outside the model.")
        return self.oe_table_base_rows + 2 * table_id + 1

    @property
    def special_token_ids(self) -> tuple[int, ...]:
        ids: list[int] = []
        try:
            for raw_range in self.special_token_scope.split(","):
                start, end = (int(value) for value in raw_range.split(":"))
                if start < 0 or end <= start:
                    raise ValueError
                ids.extend(range(start, end))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "special_token_scope must contain comma-separated start:end ranges."
            ) from exc
        return tuple(ids)
