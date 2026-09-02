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

"""Behavior guards for the temporary Flash-Lite dual model entries."""

from test.runtime.test_lite_model_loader import lite_config_dict

import pytest

from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig
from tokenspeed.runtime.configs.lite_config import LiteConfig
from tokenspeed.runtime.models.flash_kda import _canonical_flash_kda_weight_name
from tokenspeed.runtime.models.lite import LiteCheckpointLayout


def test_dual_entries_derive_the_same_model_semantics() -> None:
    raw = lite_config_dict()
    flash = FLASHLocalConfig.from_dict(raw)
    lite = LiteConfig.from_dict(raw)

    shared_fields = (
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "mla_scale_q_lora",
        "mla_scale_kv_lora",
        "mla_use_nope",
        "mla_use_output_gate",
        "n_routed_experts",
        "routed_scaling_factor",
        "moe_topk",
        "moe_switch_token_num",
        "moe_impl",
        "moe_group_size",
        "grouped_moe_norm_scale",
        "zero_expert_num",
        "zero_expert_type",
        "ngram_vocab_size_ratio",
        "emb_neighbor_num",
        "emb_split_num",
        "ngram_exclude_sp_token",
        "ngram_fix_normalize_factor",
        "fa_interval",
        "linear_method",
        "kda_nope",
        "linear_hidden_size",
        "linear_head_dim",
        "linear_num_heads",
        "linear_conv_size",
        "kda_use_full_rank_gate",
    )
    assert {name: getattr(flash, name) for name in shared_fields} == {
        name: getattr(lite, name) for name in shared_fields
    }
    assert flash.intermediate_size == lite.ffn_hidden_size
    assert flash.moe_intermediate_size == lite.expert_ffn_hidden_size
    assert flash.n_shared_experts == lite.num_shared_experts == 1
    assert flash.linear_layer_ids == lite.linear_layer_ids
    assert flash.full_attention_layer_ids == lite.full_attention_layer_ids
    assert flash.layers_block_type == lite.layers_block_type
    assert flash.oe_ignore_tokens == list(lite.special_token_ids)
    assert flash.over_embedding_m == lite.oe_table_base_rows


@pytest.mark.parametrize(
    ("source_name", "flash_target", "lite_target"),
    [
        (
            "model.layers.0.self_attn.linear_core.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.input_projection.weight",
        ),
        (
            "model.layers.0.self_attn.linear_core.b_proj.1.weight",
            "model.layers.0.self_attn.b_proj.fc2.weight",
            "model.layers.0.self_attn.beta_b_proj.weight",
        ),
        (
            "model.layers.3.self_attn.kv_b_proj.weight",
            "model.layers.3.self_attn.kv_b_proj.weight",
            "model.layers.3.self_attn.kv_b_proj.weight",
        ),
        (
            "model.layers.0.mlp.experts.1.down_proj.weight",
            "model.layers.0.moe.experts.1.down_proj.weight",
            "model.layers.0.mlp.experts.w2_weight",
        ),
        (
            "model.ngram_embeddings.embedders.0.weight",
            "model.ngram_embeddings.embedders.0.weight",
            "model.ngram_embeddings.embedders.0.weight",
        ),
    ],
)
def test_checkpoint_sources_preserve_current_physical_layouts(
    source_name: str,
    flash_target: str,
    lite_target: str,
) -> None:
    layout = LiteCheckpointLayout(LiteConfig.from_dict(lite_config_dict()))

    assert _canonical_flash_kda_weight_name(source_name) == flash_target
    assert layout.spec(source_name).target_name == lite_target
