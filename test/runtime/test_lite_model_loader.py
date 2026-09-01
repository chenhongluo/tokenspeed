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

import importlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tokenspeed.runtime.configs.lite_config import LiteConfig
from tokenspeed.runtime.models.lite import (
    FLASHLocalForCausalLM,
    LiteCheckpointLayout,
    LiteKDAParameters,
    LiteMLAParameters,
)


def lite_config_dict(**overrides):
    config = {
        "architectures": ["FLASHLocalForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "vocab_size": 120,
        "hidden_size": 96,
        "ffn_hidden_size": 96,
        "expert_ffn_hidden_size": 16,
        "num_layers": 4,
        "num_attention_heads": 8,
        "kv_lora_rank": 12,
        "q_lora_rank": 24,
        "qk_rope_head_dim": 2,
        "v_head_dim": 4,
        "qk_nope_head_dim": 4,
        "mla_scale_q_lora": True,
        "mla_scale_kv_lora": True,
        "routed_scaling_factor": 6.0,
        "n_routed_experts": 8,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "use_cache": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "zero_expert_num": 4,
        "zero_expert_type": "identity",
        "moe_topk": 2,
        "moe_switch_token_num": 16,
        "moe_impl": "mix",
        "ngram_vocab_size_ratio": 0.1,
        "emb_neighbor_num": 4,
        "emb_split_num": 4,
        "ngram_exclude_sp_token": True,
        "special_token_scope": "0:4,36:55",
        "ngram_fix_normalize_factor": True,
        "moe_group_size": 4,
        "grouped_moe_norm_scale": 2,
        "use_mla": 1,
        "attention_method": "MLA",
        "fa_interval": 4,
        "linear_method": "FGBKDA",
        "kda_nope": True,
        "linear_hidden_size": 96,
        "linear_head_dim": 4,
        "linear_num_heads": 8,
        "linear_conv_size": 4,
        "mla_use_output_gate": True,
        "kda_use_full_rank_gate": True,
    }
    config.update(overrides)
    return config


def mapping(world_size=1, rank=0, role="prefill"):
    size = world_size
    return SimpleNamespace(
        world_size=world_size,
        rank=rank,
        pp_size=1,
        dense=SimpleNamespace(tp_size=size, tp_rank=rank),
        linear_attn=SimpleNamespace(tp_size=size, tp_rank=rank),
        moe=SimpleNamespace(ep_size=size, ep_rank=rank),
        attn=SimpleNamespace(
            tp_size=1,
            cp_size=size if role == "prefill" else 1,
            dp_size=size if role == "decode" else 1,
        ),
    )


def weights(layout, mutate=None):
    for index, name in enumerate(layout.iter_source_names()):
        spec = layout.spec(name)
        tensor = torch.zeros(spec.shape, dtype=spec.dtype)
        if mutate is not None:
            tensor = mutate(index, name, tensor)
        yield name, tensor


def no_accelerator_error(exc):
    while exc is not None:
        if "requires an NVIDIA CUDA, AMD ROCm, or Ascend NPU device" in str(exc):
            return True
        exc = exc.__cause__
    return False


def test_lite_config_derives_hybrid_and_oe_geometry():
    config = LiteConfig.from_dict(lite_config_dict())

    assert config.linear_layer_ids == [0, 1, 2]
    assert config.full_attention_layer_ids == [3]
    assert config.layers_block_type == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "attention",
    ]
    assert config.oe_component_count == 12
    assert config.oe_hidden_size == 8
    assert config.oe_table_rows(0) == 13
    assert config.oe_table_rows(11) == 35
    assert config.special_token_ids == tuple(range(4)) + tuple(range(36, 55))


def test_lite_config_fails_closed_on_raw_checkpoint_fields():
    raw = lite_config_dict()
    raw.pop("linear_method")
    with pytest.raises(ValueError, match="linear_method"):
        LiteConfig.from_dict(raw)

    with pytest.raises(ValueError, match="divisible by fa_interval"):
        LiteConfig.from_dict(lite_config_dict(num_layers=5))
    with pytest.raises(ValueError, match="full-rank"):
        LiteConfig.from_dict(lite_config_dict(kda_use_full_rank_gate=False))
    with pytest.raises(ValueError, match="architectures"):
        LiteConfig.from_dict(lite_config_dict(architectures=["LlamaForCausalLM"]))


def test_get_config_selects_lite_by_architecture_without_model_type(tmp_path):
    try:
        module = importlib.import_module(
            "tokenspeed.runtime.utils.hf_transformers_utils"
        )
    except RuntimeError as exc:
        if no_accelerator_error(exc):
            pytest.skip("Full runtime import requires an accelerator platform.")
        raise
    (tmp_path / "config.json").write_text(json.dumps(lite_config_dict()))

    config = module.get_config(str(tmp_path), trust_remote_code=False)

    assert isinstance(config, LiteConfig)
    assert config.architectures == ["FLASHLocalForCausalLM"]


def test_model_registry_resolves_real_lite_entry_class():
    try:
        module = importlib.import_module("tokenspeed.runtime.models.registry")
    except RuntimeError as exc:
        if no_accelerator_error(exc):
            pytest.skip("Full runtime import requires an accelerator platform.")
        raise

    model_class, architecture = module.ModelRegistry.resolve_model_cls(
        ["FLASHLocalForCausalLM"]
    )

    assert model_class is FLASHLocalForCausalLM
    assert architecture == "FLASHLocalForCausalLM"


def test_real_layout_has_exact_source_count_and_shapes():
    config = LiteConfig()
    layout = LiteCheckpointLayout(config)
    names = list(layout.iter_source_names())

    assert len(names) == len(set(names)) == 129_870
    assert layout.spec(
        "model.layers.0.self_attn.linear_core.b_proj.1.weight"
    ).shape == (
        4096,
        128,
    )
    assert layout.spec("model.layers.3.self_attn.kv_b_proj.weight").shape == (
        8192,
        512,
    )
    assert layout.spec("model.layers.0.mlp.experts.1535.down_proj.weight").shape == (
        768,
        512,
    )
    assert layout.spec("model.ngram_embeddings.embedders.11.weight").shape == (
        4_718_615,
        256,
    )


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_model_skeleton_uses_kda_tp8_moe_ep8_and_replicated_mla(role):
    config = LiteConfig.from_dict(lite_config_dict())
    model = FLASHLocalForCausalLM(config, mapping(8, rank=3, role=role))

    assert isinstance(model.model.layers[0].self_attn, LiteKDAParameters)
    assert isinstance(model.model.layers[3].self_attn, LiteMLAParameters)
    assert model.model.layers[0].self_attn.q_proj.weight.shape == (4, 96)
    assert model.model.layers[3].self_attn.q_b_proj.weight.shape == (48, 24)
    assert tuple(model.model.layers[0].mlp.experts) == ("12", "13", "14", "15")
    assert model.model.layers[0].mlp.proj_input.weight.shape == (12, 96)
    assert model.model.layers[0].mlp.expert_groups[
        0
    ].router.classifier.weight.shape == (
        12,
        24,
    )
    assert isinstance(
        model.model.ngram_embeddings.embedders[0].weight, nn.UninitializedParameter
    )


def test_strict_loader_covers_rename_shards_experts_and_host_oe():
    config = LiteConfig.from_dict(lite_config_dict())
    model = FLASHLocalForCausalLM(config, mapping(8, rank=1))
    layout = model.layout

    def sentinel(_index, name, tensor):
        if name == "model.layers.0.self_attn.linear_core.q_proj.weight":
            return torch.arange(tensor.numel(), dtype=tensor.dtype).view(tensor.shape)
        return tensor

    loaded = model.load_weights(weights(layout, sentinel))

    source = sentinel(
        0,
        "model.layers.0.self_attn.linear_core.q_proj.weight",
        torch.empty((32, 96), dtype=torch.bfloat16),
    )
    assert torch.equal(model.model.layers[0].self_attn.q_proj.weight, source[4:8])
    assert "model.layers.0.self_attn.f_a_proj.weight" in loaded
    assert "model.layers.0.mlp.experts.4.gate_proj.weight" in loaded
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" not in loaded
    assert model.model.ngram_embeddings.embedders[0].weight.device.type == "cpu"
    assert model.model.ngram_embeddings.embedders[0].weight.shape == (13, 8)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "missing 1 source"),
        ("duplicate", "Duplicate Lite checkpoint weight"),
        ("unexpected", "Unexpected Lite checkpoint weight"),
        ("shape", "shape/dtype"),
    ],
)
def test_strict_loader_rejects_incomplete_or_ambiguous_stream(case, message):
    config = LiteConfig.from_dict(lite_config_dict())
    model = FLASHLocalForCausalLM(config, mapping())
    checkpoint = list(weights(model.layout))
    if case == "missing":
        checkpoint.pop()
    elif case == "duplicate":
        checkpoint.insert(1, checkpoint[0])
    elif case == "unexpected":
        checkpoint.insert(0, ("model.ghost.weight", torch.zeros(1)))
    else:
        name, tensor = checkpoint[0]
        checkpoint[0] = (name, tensor[:-1])

    with pytest.raises(ValueError, match=message):
        model.load_weights(checkpoint)
