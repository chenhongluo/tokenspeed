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

from tokenspeed.runtime.configs.lite_config import LiteConfig
from tokenspeed.runtime.models import lite
from tokenspeed.runtime.models.lite import (
    FLASHLocalForCausalLM,
    LiteDecoderLayer,
    LiteModel,
)


class _Mode:
    def __init__(self, idle: bool = False) -> None:
        self.idle = idle

    def is_idle(self) -> bool:
        return self.idle

    def is_decode(self) -> bool:
        return not self.idle

    def is_extend_or_mixed(self) -> bool:
        return False


class _Capture:
    @staticmethod
    def need_capture() -> bool:
        return False


class _Scale(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
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
        self.global_sp_num_tokens = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        global_sp_num_tokens: list[int] | None = None,
    ) -> torch.Tensor:
        self.global_sp_num_tokens = global_sp_num_tokens
        return hidden_states + self.offset


def _ctx(*, idle: bool = False, global_num_tokens=None):
    return SimpleNamespace(
        forward_mode=_Mode(idle),
        global_num_tokens=global_num_tokens,
        capture_hidden_mode=_Capture(),
        gather_ids=None,
    )


def _fake_layer(config, role: str, layer_id: int) -> LiteDecoderLayer:
    layer = LiteDecoderLayer(config, mapping(8, rank=3, role=role), layer_id)
    layer.input_layernorm = _Scale(2)
    layer.self_attn = _Attention(1)
    layer.post_attention_layernorm = _Scale(3)
    layer.mlp = _MLP(-2)
    return layer


def test_decoder_matches_double_residual_oracle() -> None:
    config = LiteConfig.from_dict(lite_config_dict())
    layer = LiteDecoderLayer(config, mapping(), 0)
    layer.input_layernorm = _Scale(2)
    layer.self_attn = _Attention(1)
    layer.post_attention_layernorm = _Scale(3)
    layer.mlp = _MLP(-2)
    hidden = torch.arange(12, dtype=torch.float32).view(3, 4)

    actual = layer(torch.arange(3), hidden, _ctx(), torch.arange(3))

    assert torch.equal(actual, 12 * hidden + 2)
    assert layer.mlp.global_sp_num_tokens is None


def test_kda_reduces_once_and_prefill_passes_token_split(monkeypatch) -> None:
    config = LiteConfig.from_dict(lite_config_dict())
    layer = _fake_layer(config, "prefill", 0)
    calls = []

    def fake_all_reduce(value, group):
        calls.append((value.clone(), group))
        return value + 10

    monkeypatch.setattr(lite, "_all_reduce", fake_all_reduce)
    hidden = torch.ones(2, 4)
    split = [2, 0, 3, 2, 1, 0, 4, 1]

    actual = layer(
        torch.arange(2), hidden, _ctx(global_num_tokens=split), torch.arange(2)
    )

    assert len(calls) == 1
    assert calls[0][1] == tuple(range(8))
    assert layer.mlp.global_sp_num_tokens is split
    assert torch.equal(actual, torch.full_like(hidden, 54))


@pytest.mark.parametrize(
    ("role", "expected_split"), [("prefill", True), ("decode", False)]
)
def test_mla_skips_kda_reduce(monkeypatch, role, expected_split) -> None:
    config = LiteConfig.from_dict(lite_config_dict())
    layer = _fake_layer(config, role, 3)
    monkeypatch.setattr(
        lite,
        "_all_reduce",
        lambda *_args: pytest.fail("MLA must not use the KDA output reduction"),
    )
    split = [1] * 8

    layer(
        torch.arange(1),
        torch.ones(1, 4),
        _ctx(global_num_tokens=split),
        torch.arange(1),
    )

    assert (layer.mlp.global_sp_num_tokens is split) is expected_split


def test_idle_layer_is_an_exact_noop() -> None:
    config = LiteConfig.from_dict(lite_config_dict())
    layer = _fake_layer(config, "decode", 0)
    hidden = torch.randn(2, 4)

    output = layer(torch.arange(2), hidden, _ctx(idle=True), torch.arange(2))

    assert output is hidden


def test_model_merges_prepared_oe_before_layers() -> None:
    config = LiteConfig.from_dict(lite_config_dict())
    model = LiteModel(config, mapping())
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
    config = LiteConfig.from_dict(lite_config_dict())
    model = FLASHLocalForCausalLM(config, mapping())
    hidden = torch.arange(192, dtype=torch.bfloat16).view(2, 96)

    class FakeModel(nn.Module):
        def forward(self, *_args):
            return hidden, None

    class FakeLogits(nn.Module):
        def forward(self, input_ids, states, head, metadata):
            assert torch.equal(input_ids, torch.tensor([5, 6]))
            assert states is hidden
            assert head is model.lm_head
            assert metadata == "metadata"
            return states

    model.model = FakeModel()
    model.logits_processor = FakeLogits()
    monkeypatch.setattr(lite, "_logits_metadata", lambda _ctx: "metadata")

    output = model(_ctx(), torch.tensor([5, 6]), torch.arange(2), torch.arange(2))

    assert output is hidden


def test_decode_embedding_and_head_follow_dense_tp8() -> None:
    config = LiteConfig.from_dict(lite_config_dict())
    model = FLASHLocalForCausalLM(config, mapping(8, rank=3, role="decode"))

    assert model.model.embed_tokens.weight.shape == (15, 96)
    assert model.model.embed_tokens.tp_rank == 3
    assert model.model.embed_tokens.tp_group == tuple(range(8))
    assert model.lm_head.weight.shape == (15, 96)


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
    config = LiteConfig.from_dict(lite_config_dict())
    layer = LiteDecoderLayer(config, mapping(), 0)
    layer.self_attn = _Attention(0.25)
    layer.mlp = _MLP(-0.5)
    layer.input_layernorm.weight.data.fill_(1)
    layer.post_attention_layernorm.weight.data.fill_(1)
    layer = layer.to("npu:0")
    hidden = torch.arange(192, dtype=torch.bfloat16, device="npu:0").view(2, 96)

    output = layer(
        torch.arange(2, device="npu:0"),
        hidden,
        _ctx(),
        torch.arange(2, device="npu:0"),
    )
    hidden_cpu = hidden.cpu()
    attention = _rms_oracle(hidden_cpu, config.rms_norm_eps) + 0.25
    residual = hidden_cpu + attention
    expected = residual + _rms_oracle(residual, config.rms_norm_eps) - 0.5

    assert torch.isfinite(output).all().item()
    assert output.shape == hidden.shape
    assert torch.allclose(output.cpu().float(), expected.float(), atol=0.02, rtol=0.02)


@pytest.mark.skipif(not _npu_available(), reason="requires an Ascend NPU")
def test_oe_embedding_norm_and_logits_on_npu() -> None:
    config = LiteConfig.from_dict(lite_config_dict())
    model = FLASHLocalForCausalLM(config, mapping())
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
    normalized = _rms_oracle(word, config.rms_norm_eps)
    expected = normalized @ torch.full((96, 120), 0.01, dtype=torch.bfloat16)

    assert output.next_token_logits.shape == (2, 120)
    assert torch.isfinite(output.next_token_logits).all().item()
    assert torch.allclose(
        output.next_token_logits.cpu().float(),
        expected.float(),
        atol=0.02,
        rtol=0.02,
    )
