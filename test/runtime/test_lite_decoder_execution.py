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

from test.runtime.test_lite_model_loader import lite_config_dict, mapping
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.models.flash_kda import (
    FLASHLocalDecoderLayer,
    FLASHLocalForCausalLM,
    FLASHLocalModel,
)
from tokenspeed.runtime.models.flash_local_moe import PackedRMSNorm


class _Mode:
    def __init__(self, idle: bool = False, extend: bool = False) -> None:
        self.idle = idle
        self.extend = extend

    def is_idle(self) -> bool:
        return self.idle

    def is_decode(self) -> bool:
        return not self.idle and not self.extend

    def is_extend_or_mixed(self) -> bool:
        return self.extend


class _Capture:
    @staticmethod
    def need_capture() -> bool:
        return False


class _Scale(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.residual_calls = 0

    def forward(
        self, value: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            self.residual_calls += 1
            residual = residual + value
            return residual * self.scale, residual
        return value * self.scale


class _Attention(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, **kwargs) -> torch.Tensor:
        return kwargs["hidden_states"] + self.offset


class _MLP(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset
        self.ctx = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx=None,
        **_kwargs,
    ) -> torch.Tensor:
        self.ctx = ctx
        return hidden_states + self.offset


def _ctx(*, idle: bool = False, extend: bool = False, global_num_tokens=None):
    return SimpleNamespace(
        forward_mode=_Mode(idle, extend),
        global_num_tokens=global_num_tokens,
        input_num_tokens=1,
        collective_num_tokens=None,
        collective_global_num_tokens=None,
        capture_hidden_mode=_Capture(),
        gather_ids=None,
    )


def test_decoder_matches_double_residual_oracle() -> None:
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    layer = FLASHLocalDecoderLayer(config, 0, Mapping(rank=0))
    layer.input_layernorm = _Scale(2)
    layer.self_attn = _Attention(1)
    layer.post_attention_layernorm = _Scale(3)
    layer.moe = _MLP(-2)
    layer.moe_comm.input_layernorm = layer.input_layernorm
    layer.moe_comm.post_attn_layernorm = layer.post_attention_layernorm
    hidden = torch.arange(12, dtype=torch.float32).view(3, 4)

    actual, residual = layer(
        torch.arange(3), hidden, _ctx(), torch.arange(3), residual=None
    )

    assert torch.equal(actual + residual, 12 * hidden + 2)
    assert layer.input_layernorm.residual_calls == 0
    assert layer.post_attention_layernorm.residual_calls == 1
    assert layer.moe.ctx.forward_mode.is_decode()


def test_packed_moe_owns_its_output_collective(monkeypatch) -> None:
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    layer = FLASHLocalDecoderLayer(config, 0, Mapping(rank=0))
    hidden = torch.zeros(2, 96, dtype=torch.bfloat16)
    residual = torch.ones_like(hidden)
    expected = torch.full_like(hidden, 3)
    monkeypatch.setattr(layer.moe, "forward", lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(
        layer.moe_comm,
        "post_mlp_fused",
        lambda *_args, **_kwargs: pytest.fail(
            "packed Grouped MoE must not be reduced a second time"
        ),
    )

    output, residual_out = layer._post_moe(expected, residual, _ctx())

    assert output is expected
    assert residual_out is residual


def test_lite_norm_preserves_fused_residual_semantics_on_cpu() -> None:
    norm = PackedRMSNorm(4)
    norm.weight.data.fill_(1)
    value = torch.tensor([[1.5, -2.0, 3.0, -4.5]], dtype=torch.bfloat16)
    residual = torch.tensor([[2.0, 1.0, -1.5, 0.5]], dtype=torch.bfloat16)
    summed = value.float() + residual.float()
    expected_residual = summed.to(torch.bfloat16)
    expected = (summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)).to(
        torch.bfloat16
    )

    output, residual_out = norm(value, residual)

    assert residual_out is residual
    assert torch.equal(residual, expected_residual)
    assert torch.equal(output, expected)


def test_kda_uses_linear_attention_mapping() -> None:
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    parallel = mapping(8, rank=3, role="prefill")
    layer = FLASHLocalDecoderLayer(config, 0, parallel)

    assert layer.moe_comm.attn_mapping is parallel.linear_attn


def test_mla_uses_mla_attention_mapping() -> None:
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    parallel = mapping(8, rank=3, role="prefill")
    layer = FLASHLocalDecoderLayer(config, 3, parallel)

    assert layer.moe_comm.attn_mapping is parallel.attn


def test_replicated_mla_owns_its_output_collective(monkeypatch) -> None:
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    layer = FLASHLocalDecoderLayer(config, 3, Mapping(rank=0))
    layer.post_attention_layernorm = _Scale(2)
    hidden = torch.zeros(2, 96, dtype=torch.bfloat16)
    residual = torch.ones_like(hidden)
    monkeypatch.setattr(
        layer.moe_comm,
        "post_attn_comm",
        lambda *_args, **_kwargs: pytest.fail(
            "replicated MLA output must not be reduced a second time"
        ),
    )

    output, residual_out = layer._post_attn(hidden, residual, _ctx())

    assert output.shape == hidden.shape
    assert torch.equal(residual_out, residual)


def test_idle_layer_keeps_graph_shape_without_attention() -> None:
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    layer = FLASHLocalDecoderLayer(config, 0, Mapping(rank=0))
    layer.self_attn = nn.Module()
    layer.input_layernorm = _Scale(2)
    layer.moe_comm.input_layernorm = layer.input_layernorm
    layer.moe = _MLP(-2)
    hidden = torch.randn(2, 4)

    output, residual = layer(
        torch.arange(2), hidden, _ctx(idle=True), torch.arange(2), residual=None
    )

    assert torch.equal(output, hidden * 2 - 2)
    assert residual is hidden


def test_model_merges_prepared_oe_before_layers() -> None:
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    model = FLASHLocalModel(config, mapping(), oe_table_placement="host")
    model.layers = nn.ModuleList()
    model.norm = nn.Identity()
    model.embed_tokens.weight.zero_()
    model.embed_tokens.weight[2].fill_(2)
    model.embed_tokens.weight[60].fill_(13**0.5)
    model.ngram_embeddings.projection.zero_()
    raw = torch.zeros(2, 12, 8, dtype=torch.bfloat16)
    model.ngram_embeddings._runtime = SimpleNamespace(
        prepared_raw_oe=lambda num_tokens: raw[:num_tokens]
    )

    output, auxiliary = model(
        torch.tensor([2, 60]), torch.arange(2), _ctx(), torch.arange(2)
    )

    assert auxiliary is None
    assert torch.equal(output[0], torch.full((96,), 2, dtype=torch.bfloat16))
    assert torch.allclose(output[1].float(), torch.ones(96), atol=0.02, rtol=0)


def test_causal_wrapper_uses_dense_head_and_shared_logits_processor(
    monkeypatch,
) -> None:
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    model = FLASHLocalForCausalLM(config, Mapping(rank=0), oe_table_placement="host")
    hidden = torch.arange(192, dtype=torch.bfloat16).view(2, 96)

    class FakeModel(nn.Module):
        def forward(self, *_args):
            return hidden, None

    class FakeLogits(nn.Module):
        def forward(self, input_ids, states, head, metadata, aux_hidden_states):
            assert torch.equal(input_ids, torch.tensor([5, 6]))
            assert states is hidden
            assert head is model.lm_head
            assert metadata == "metadata"
            assert aux_hidden_states is None
            return states

    model.model = FakeModel()
    model.logits_processor = FakeLogits()
    monkeypatch.setattr(
        "tokenspeed.runtime.layers.logits_processor.LogitsMetadata.from_forward_context",
        lambda _ctx: "metadata",
    )

    output = model(_ctx(), torch.tensor([5, 6]), torch.arange(2), torch.arange(2))

    assert output is hidden


def test_decode_embedding_and_head_follow_dense_tp8() -> None:
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    model = FLASHLocalForCausalLM(
        config,
        Mapping(
            rank=3,
            world_size=8,
            attn_tp_size=1,
            attn_cp_size=1,
            attn_dp_size=8,
            dense_tp_size=8,
            dense_dp_size=1,
            moe_tp_size=1,
            moe_ep_size=8,
            moe_dp_size=1,
            linear_attn_tp_size=8,
            mla_weight_tp_size=1,
        ),
        oe_table_placement="host",
    )

    assert model.model.embed_tokens.weight.shape == (16, 96)
    assert model.model.embed_tokens.tp_rank == 3
    assert model.model.embed_tokens.tp_group == tuple(range(8))
    assert model.lm_head.weight.shape == (16, 96)


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return torch.npu.is_available()


def _rms_oracle(value: torch.Tensor, eps: float) -> torch.Tensor:
    value_fp32 = value.float()
    return (
        value_fp32 * torch.rsqrt(value_fp32.square().mean(dim=-1, keepdim=True) + eps)
    ).to(value.dtype)


@pytest.mark.skipif(not _npu_available(), reason="requires an Ascend NPU")
def test_decoder_residual_path_on_npu() -> None:
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    layer = FLASHLocalDecoderLayer(config, 0, Mapping(rank=0))
    layer.self_attn = _Attention(0.25)
    layer.moe = _MLP(-0.5)
    layer.input_layernorm.weight.data.fill_(1)
    layer.post_attention_layernorm.weight.data.fill_(1)
    layer = layer.to(device="npu:0", dtype=torch.bfloat16)
    hidden = torch.arange(192, dtype=torch.bfloat16, device="npu:0").view(2, 96)
    hidden_before = hidden.cpu().clone()

    assert layer.post_attention_layernorm.weight.dtype == torch.bfloat16

    output, residual_out = layer(
        torch.arange(2, device="npu:0"),
        hidden,
        _ctx(),
        torch.arange(2, device="npu:0"),
        residual=None,
    )
    attention = _rms_oracle(hidden_before, config.rms_norm_eps) + 0.25
    residual_fp32 = hidden_before.float() + attention.float()
    residual = residual_fp32.to(torch.bfloat16)
    moe_input = (
        residual_fp32
        * torch.rsqrt(
            residual_fp32.square().mean(dim=-1, keepdim=True) + config.rms_norm_eps
        )
    ).to(torch.bfloat16)
    expected = residual + moe_input - 0.5

    output = output + residual_out
    assert torch.isfinite(output).all().item()
    assert output.shape == hidden.shape
    assert torch.equal(hidden.cpu(), residual)
    assert torch.allclose(output.cpu().float(), expected.float(), atol=0.02, rtol=0.02)


@pytest.mark.skipif(not _npu_available(), reason="requires an Ascend NPU")
def test_oe_embedding_and_logits_on_npu() -> None:
    config = FLASHLocalConfig.from_dict(lite_config_dict())
    model = FLASHLocalForCausalLM(config, Mapping(rank=0), oe_table_placement="host")
    model.model.layers = nn.ModuleList()
    model = model.to("npu:0")
    with torch.no_grad():
        model.model.embed_tokens.weight.zero_()
        model.model.embed_tokens.weight[2].fill_(2)
        model.model.embed_tokens.weight[60].fill_(13**0.5)
        model.model.ngram_embeddings.projection.zero_()
        model.model.norm.weight.fill_(1)
        model.lm_head.weight.fill_(0.01)
    raw = torch.zeros(2, 12, 8, dtype=torch.bfloat16, device="npu:0")
    model.model.ngram_embeddings._runtime = SimpleNamespace(
        prepared_raw_oe=lambda num_tokens: raw[:num_tokens]
    )
    input_ids = torch.tensor([2, 60], device="npu:0")

    output = model(
        _ctx(),
        input_ids,
        torch.arange(2, device="npu:0"),
        torch.arange(2, device="npu:0"),
    )
    word = torch.stack(
        (
            torch.full((96,), 2, dtype=torch.bfloat16),
            torch.full((96,), 13**0.5, dtype=torch.bfloat16) / (13**0.5),
        )
    )
    expected = word @ torch.full((96, 120), 0.01, dtype=torch.bfloat16)

    assert output.next_token_logits.shape == (2, 120)
    assert torch.isfinite(output.next_token_logits).all().item()
    assert torch.allclose(
        output.next_token_logits.cpu().float(),
        expected.float(),
        atol=0.02,
        rtol=0.02,
    )
