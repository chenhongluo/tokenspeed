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

import os
import subprocess
import sys
from pathlib import Path
from test.runtime.test_lite_model_loader import lite_config_dict, mapping, weights
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from tokenspeed.runtime.configs.lite_config import LiteConfig
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.models.flash_local_moe import (
    PackedFLASHLocalMoE,
    PackedWeight,
    grouped_moe_local_expert_ids,
    grouped_moe_topk,
)
from tokenspeed.runtime.models.lite import FLASHLocalForCausalLM


def test_lite_packed_moe_import_does_not_require_accelerator() -> None:
    root = Path(__file__).parents[2]
    env = os.environ.copy()
    source_paths = [root / "python", root / "tokenspeed-kernel" / "python"]
    env["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in source_paths), env.get("PYTHONPATH", "")]
    )
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import tokenspeed.runtime.layers.moe.loader; "
            "import tokenspeed.runtime.layers.moe.weights.unquant; "
            "print('tokenspeed_kernel' in sys.modules)",
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert probe.stdout.strip() == "False"


def test_lite_router_bias_only_changes_selection() -> None:
    config = LiteConfig.from_dict(lite_config_dict(routed_scaling_factor=5.0))
    layer = PackedFLASHLocalMoE(config, mapping())
    grouped = torch.randn(3, config.moe_group_size, 24, dtype=torch.bfloat16)
    for group_id, expert_group in enumerate(layer.expert_groups):
        expert_group.router.classifier.weight.data.zero_()
        expert_group.router.e_score_correction_bias.data.copy_(
            torch.arange(12, dtype=torch.float32) * (group_id + 1)
        )

    topk_weights, topk_ids, logits = layer._route(grouped)

    probabilities = torch.softmax(logits, dim=-1)
    for group_id, expert_group in enumerate(layer.expert_groups):
        expected_ids = torch.topk(
            probabilities[group_id] + expert_group.router.e_score_correction_bias,
            2,
            dim=-1,
            sorted=False,
        ).indices
        assert torch.equal(topk_ids[group_id], expected_ids.to(torch.int32))
    assert torch.allclose(
        topk_weights,
        torch.full_like(topk_weights, 5.0 / 12.0),
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        topk_weights.sum(dim=-1),
        torch.full_like(topk_weights.sum(dim=-1), 10.0 / 12.0),
    )


def test_grouped_moe_shared_topk_and_ep8_ownership_contract() -> None:
    logits = torch.tensor([[[0.0, 2.0, 1.0]], [[2.0, 0.0, 1.0]]], dtype=torch.float32)
    bias = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])

    weights, ids = grouped_moe_topk(
        logits,
        bias,
        top_k=1,
        renormalize=False,
        routed_scaling_factor=2.0,
    )

    assert ids.tolist() == [[[0]], [[1]]]
    assert ids.dtype == torch.int64
    expected = logits.softmax(dim=-1).gather(-1, ids.to(torch.int64)) * 2
    torch.testing.assert_close(weights, expected)
    ownership = [
        grouped_moe_local_expert_ids(
            num_groups=4,
            num_experts_per_group=384,
            ep_rank=rank,
            ep_size=8,
        )
        for rank in range(8)
    ]
    assert all(len(ids) == 192 for ids in ownership)
    assert sorted(expert for rank_ids in ownership for expert in rank_ids) == list(
        range(1536)
    )
    assert all(
        sum(group * 384 <= expert < (group + 1) * 384 for expert in rank_ids) == 48
        for rank_ids in ownership
        for group in range(4)
    )


def test_packed_weight_loader_uses_target_owned_dense_shard() -> None:
    layer = PackedWeight(
        (2, 4),
        torch.bfloat16,
        shard_axis=0,
        shard_rank=1,
        shard_size=2,
    )
    source = torch.arange(16, dtype=torch.bfloat16).view(4, 4)

    layer.weight.weight_loader(layer.weight, source)

    torch.testing.assert_close(layer.weight, source[2:])


def test_shared_flash_local_loader_uses_packed_group_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tokenspeed.runtime.configs.flash_kda_config import FLASHLocalConfig
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models import flash_kda

    monkeypatch.setattr(
        flash_kda, "flash_local_prefers_packed_projections", lambda: True
    )
    monkeypatch.setattr(flash_kda, "flash_local_prefers_packed_moe", lambda: True)
    config = FLASHLocalConfig.from_dict(lite_config_dict(ngram_vocab_size_ratio=None))
    parallel = Mapping(
        rank=1,
        world_size=8,
        attn_tp_size=1,
        attn_cp_size=8,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=8,
        linear_attn_tp_size=8,
        mla_weight_tp_size=1,
    )
    model = flash_kda.FLASHLocalForCausalLM(config, parallel)
    prefix = "model.layers.0.mlp.experts.1"

    model.load_weights(
        [
            (
                f"{prefix}.gate_proj.weight",
                torch.full((16, 24), 1, dtype=torch.bfloat16),
            ),
            (
                f"{prefix}.up_proj.weight",
                torch.full((16, 24), 2, dtype=torch.bfloat16),
            ),
            (
                f"{prefix}.down_proj.weight",
                torch.full((24, 16), 3, dtype=torch.bfloat16),
            ),
        ]
    )

    experts = model.model.layers[0].moe.experts
    assert model.model.layers[0].moe.local_expert_ids == (1, 9, 17, 25)
    assert torch.equal(
        experts.w13_weight[0, :16], torch.ones_like(experts.w13_weight[0, :16])
    )
    assert torch.equal(
        experts.w13_weight[0, 16:], torch.full_like(experts.w13_weight[0, 16:], 2)
    )
    assert torch.equal(experts.w2_weight[0], torch.full_like(experts.w2_weight[0], 3))


def _grouped_moe_oracle(
    layer: PackedFLASHLocalMoE,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    config = layer.config
    projected = F.linear(hidden_states, layer.proj_input.weight)
    projected_fp32 = projected.float()
    projected = (
        projected_fp32
        * torch.rsqrt(
            projected_fp32.square().mean(dim=-1, keepdim=True) + config.rms_norm_eps
        )
        * layer.norm.weight.float()
    ).to(projected.dtype)
    grouped = (projected * config.grouped_moe_norm_scale).view(
        hidden_states.shape[0], config.moe_group_size, layer.group_hidden
    )
    outputs = torch.zeros_like(grouped)
    for group_id in range(config.moe_group_size):
        for token_id in range(hidden_states.shape[0]):
            for route_id in range(config.moe_topk):
                expert_id = int(topk_ids[group_id, token_id, route_id])
                weight = topk_weights[group_id, token_id, route_id].to(grouped.dtype)
                if expert_id >= config.n_routed_experts:
                    outputs[token_id, group_id] += grouped[token_id, group_id] * weight
                    continue
                flat_id = group_id * config.n_routed_experts + expert_id
                gate_up = F.linear(
                    grouped[token_id, group_id], layer.experts.w13_weight[flat_id]
                )
                gate, up = gate_up.chunk(2)
                outputs[token_id, group_id] += (
                    F.linear(F.silu(gate) * up, layer.experts.w2_weight[flat_id])
                    * weight
                )
    routed = F.linear(outputs.flatten(1), layer.proj_output.weight)
    shared = F.linear(
        F.silu(F.linear(hidden_states, layer.shared_experts.gate_proj.weight))
        * F.linear(hidden_states, layer.shared_experts.up_proj.weight),
        layer.shared_experts.down_proj.weight,
    )
    return routed + shared


def test_lite_grouped_moe_reference_real_identity_duplicate_and_shared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(7)
    config = LiteConfig.from_dict(lite_config_dict())
    layer = PackedFLASHLocalMoE(config, mapping())
    for parameter in layer.parameters():
        parameter.data.uniform_(-0.08, 0.08)
    hidden_states = torch.randn(3, 96, dtype=torch.bfloat16)
    ids_one_group = torch.tensor([[0, 0], [8, 9], [1, 8]], dtype=torch.int32)
    topk_ids = torch.stack([ids_one_group.roll(group_id, 0) for group_id in range(4)])
    topk_weights = torch.tensor(
        [[[0.25, 0.5], [0.4, 0.3], [0.2, 0.6]]] * 4,
        dtype=torch.float32,
    )
    logits = torch.zeros(4, 3, 12, dtype=torch.float32)
    monkeypatch.setattr(
        layer, "_route", lambda _grouped: (topk_weights, topk_ids, logits)
    )

    actual = layer(hidden_states)
    expected = _grouped_moe_oracle(layer, hidden_states, topk_weights, topk_ids)

    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_lite_grouped_moe_empty_and_prefill_metadata_fail_closed() -> None:
    config = LiteConfig.from_dict(lite_config_dict())
    reference = PackedFLASHLocalMoE(config, mapping())
    empty = torch.empty(0, 96, dtype=torch.bfloat16)
    assert reference(empty).shape == empty.shape

    distributed = PackedFLASHLocalMoE(config, mapping(8, rank=3))
    ctx = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        global_num_tokens=None,
    )
    with pytest.raises(ValueError, match="global_num_tokens"):
        distributed(torch.zeros(1, 96, dtype=torch.bfloat16), ctx=ctx)


def _identity_route(layer: PackedFLASHLocalMoE, tokens: int):
    weights = torch.ones(4, tokens, 2, dtype=torch.float32)
    ids = torch.full((4, tokens, 2), 8, dtype=torch.int32)
    logits = torch.zeros(4, tokens, 12, dtype=torch.float32)
    return weights, ids, logits


def _stub_comm_ops(monkeypatch: pytest.MonkeyPatch, **operations):
    distributed = ModuleType("tokenspeed.runtime.distributed")
    comm_ops = ModuleType("tokenspeed.runtime.distributed.comm_ops")
    for name, operation in operations.items():
        setattr(comm_ops, name, operation)
    distributed.comm_ops = comm_ops
    monkeypatch.setitem(sys.modules, distributed.__name__, distributed)
    monkeypatch.setitem(sys.modules, comm_ops.__name__, comm_ops)


def test_lite_grouped_moe_prefill_collectives_and_identity_math(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(11)
    config = LiteConfig.from_dict(lite_config_dict())
    layer = PackedFLASHLocalMoE(config, mapping(8, rank=0, role="prefill"))
    for parameter in layer.parameters():
        parameter.data.uniform_(-0.05, 0.05)
    hidden = torch.randn(2, 96, dtype=torch.bfloat16)
    monkeypatch.setattr(
        layer, "_route", lambda grouped: _identity_route(layer, len(grouped))
    )
    calls = []

    def gather(tensor, _group, _split):
        calls.append(("ag", tuple(tensor.shape)))
        return tensor.repeat((8,) + (1,) * (tensor.ndim - 1))

    def scatter(tensor, _group, split):
        calls.append(("rs", tuple(tensor.shape)))
        return tensor[: split[0]] * 8

    _stub_comm_ops(monkeypatch, token_all_gather=gather, token_reduce_scatter=scatter)

    actual = layer(
        hidden,
        ctx=SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            global_num_tokens=[2] * 8,
        ),
    )
    grouped = layer._project_grouped(hidden)
    expected = F.linear(
        grouped.flatten(1) * 2, layer.proj_output.weight
    ) + layer._shared(hidden)

    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)
    assert calls == [
        ("ag", (2, 96)),
        ("ag", (2, 8)),
        ("ag", (2, 8)),
        ("rs", (16, 96)),
    ]


def test_lite_grouped_moe_decode_feature_expert_and_dense_collectives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(13)
    config = LiteConfig.from_dict(lite_config_dict())
    layer = PackedFLASHLocalMoE(config, mapping(8, rank=0, role="decode"))
    for parameter in layer.parameters():
        parameter.data.uniform_(-0.05, 0.05)
    hidden = torch.randn(2, 96, dtype=torch.bfloat16)
    monkeypatch.setattr(
        layer, "_route", lambda grouped: _identity_route(layer, len(grouped))
    )
    calls = []

    def gather(tensor, _group, dim=-1):
        calls.append(("ag", dim, tuple(tensor.shape)))
        return tensor.repeat(1, 8)

    def reduce(tensor, _group):
        calls.append(("ar", tuple(tensor.shape)))
        return tensor * 8

    _stub_comm_ops(monkeypatch, all_gather=gather, all_reduce=reduce)

    actual = layer(
        hidden,
        ctx=SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            global_num_tokens=[2] * 8,
        ),
    )
    local_projection = F.linear(hidden, layer.proj_input.weight)
    projected = local_projection.repeat(1, 8)
    projected_fp32 = projected.float()
    grouped = (
        (
            projected_fp32
            * torch.rsqrt(
                projected_fp32.square().mean(dim=-1, keepdim=True) + config.rms_norm_eps
            )
            * layer.norm.weight.float()
        ).to(projected.dtype)
        * config.grouped_moe_norm_scale
    ).view(2, 4, 24)
    routed_local = (grouped.flatten(1) * 2)[:, :12]
    expected = (
        F.linear(routed_local, layer.proj_output.weight) + layer._shared(hidden)
    ) * 8

    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)
    assert calls == [
        ("ag", -1, (2, 12)),
        ("ar", (2, 96)),
        ("ar", (2, 96)),
    ]


@pytest.mark.parametrize("world_size", [1, 8])
def test_lite_grouped_moe_meta_packed_shapes_and_loader(world_size: int) -> None:
    config = LiteConfig.from_dict(lite_config_dict())
    with torch.device("meta"):
        model = FLASHLocalForCausalLM(config, mapping(world_size, rank=0))
    experts = model.model.layers[0].mlp.experts
    local_experts = 32 // world_size
    assert experts.w13_weight.shape == (local_experts, 32, 24)
    assert experts.w2_weight.shape == (local_experts, 24, 16)
    assert experts.w13_weight.is_meta and experts.w2_weight.is_meta

    loaded = model.load_weights(weights(model.layout))
    assert "model.layers.0.mlp.experts.w13_weight" in loaded
    assert "model.layers.0.mlp.experts.w2_weight" in loaded
