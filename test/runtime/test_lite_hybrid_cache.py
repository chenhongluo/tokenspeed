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

from collections import Counter
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tokenspeed.runtime.configs.lite_config import LiteConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)

_ARCHITECTURE = "FLASHLocalForCausalLM"
_STATE_GROUPS = tuple(f"{LINEAR_ATTENTION}_{index}" for index in range(3))


class _OperationMetadata:
    def __init__(self, tables, forward_op):
        self.tables = tables
        self.forward_op = forward_op

    def require_table(self, group_id, *, active_forward_op):
        if active_forward_op is not self.forward_op:
            raise RuntimeError("stale metadata")
        return self.tables[group_id]


def _lite_recipe(
    tp_size: int,
    *,
    device: str = "cpu",
    max_bs: int = 8,
    token_limit: int = 4096,
    overlap_schedule_depth: int = 0,
) -> Any:
    try:
        from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
        from tokenspeed.runtime.layers.attention.configs.linear_attn import (
            LinearAttnConfig,
        )
        from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.lite import LiteRecipe
    except ModuleNotFoundError as exc:
        if exc.name == "compressed_tensors":
            pytest.skip("full attention config dependencies are not installed")
        raise

    text_config = LiteConfig()
    text_config.architectures = [_ARCHITECTURE]
    linear = LinearAttnConfig(
        num_k_heads=text_config.linear_num_heads,
        num_v_heads=text_config.linear_num_heads,
        head_k_dim=text_config.linear_head_dim,
        head_v_dim=text_config.linear_head_dim,
        conv_kernel_size=text_config.linear_conv_size,
        layer_ids=tuple(text_config.linear_layer_ids),
        tp_size=tp_size,
    )
    mla = MLAConfig(
        backend_name="mla",
        num_attention_heads=text_config.num_attention_heads,
        num_kv_heads=text_config.num_key_value_heads,
        head_dim=text_config.qk_nope_head_dim + text_config.qk_rope_head_dim,
        attn_tp_size=1,
        kv_lora_rank=text_config.kv_lora_rank,
        qk_nope_head_dim=text_config.qk_nope_head_dim,
        qk_rope_head_dim=text_config.qk_rope_head_dim,
        v_head_dim=text_config.v_head_dim,
        scaling=(text_config.qk_nope_head_dim + text_config.qk_rope_head_dim) ** -0.5,
        kv_cache_dim=text_config.kv_lora_rank + text_config.qk_rope_head_dim,
    )
    attn_config = AttnConfig(
        device=device,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method=None,
        prefix_granularity=128,
        max_bs=max_bs,
        max_graph_bs=max_bs,
        context_len=token_limit,
        pd_disaggregation_enabled=True,
        speculative_num_draft_tokens=1,
        components=(mla, linear),
    )
    return LiteRecipe(
        server_args=SimpleNamespace(
            max_total_tokens=token_limit,
            chunked_prefill_size=128,
            speculative_algorithm=None,
            speculative_num_draft_tokens=1,
        ),
        model_config=SimpleNamespace(hf_config=text_config),
        attn_config=attn_config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=1 << 40,
        decode_input_tokens=1,
        overlap_schedule_depth=overlap_schedule_depth,
    )


def _layout(tp_size: int, **recipe_kwargs):
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack

    recipe = _lite_recipe(tp_size, **recipe_kwargs)
    groups = recipe.groups()
    layout = pack(
        groups,
        prefix_granularity=recipe.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(groups),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )
    recipe.check_layout(layout)
    return recipe, groups, layout


def _pool(device: str = "cpu", *, num_lcm_blocks: int = 8):
    try:
        from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
        from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
            HybridKDATokenToKVPool,
        )
    except RuntimeError as exc:
        if "requires an NVIDIA CUDA, AMD ROCm, or Ascend NPU device" in str(exc):
            pytest.skip("cache arena runtime requires an accelerator platform")
        raise

    recipe, groups, layout = _layout(8, device=device)
    plan = layout.bind(num_lcm_blocks)
    arena = CacheArena(
        plan,
        device,
        cache_group_specs=tuple(spec for spec, _ in groups),
        token_capacity=512,
    )
    pool = HybridKDATokenToKVPool(
        arena=arena,
        dtype=torch.bfloat16,
        model_dtype=torch.bfloat16,
        quant_method=None,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=28,
        rank=0,
        layer_types=recipe.layer_types,
    )
    return recipe, pool


def _accelerator_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    pytest.skip("requires CUDA or Ascend NPU")


def test_lite_is_registered_as_the_existing_hybrid_mla_kda_family() -> None:
    try:
        from tokenspeed.runtime.layers.attention.configs.linear_attn import (
            LinearAttnConfig,
        )
        from tokenspeed.runtime.layers.attention.registry import (
            _HYBRID_MLA_KDA_ARCHITECTURES,
            _LINEAR_ATTN_CLS,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "compressed_tensors":
            pytest.skip("full attention registry dependencies are not installed")
        raise
    except RuntimeError as exc:
        if "requires an NVIDIA CUDA, AMD ROCm, or Ascend NPU device" in str(exc):
            pytest.skip("full attention registry requires an accelerator")
        raise

    assert _ARCHITECTURE in _HYBRID_MLA_KDA_ARCHITECTURES
    assert _LINEAR_ATTN_CLS[_ARCHITECTURE] is LinearAttnConfig


@pytest.mark.parametrize(
    (
        "tp_size",
        "conv_shape",
        "recurrent_shape",
        "packing",
        "plane_bytes",
        "parent_bytes",
    ),
    (
        (8, (1536, 3), (4, 128, 128), (2, 1), 294_912, 2_064_384),
        (1, (12288, 3), (32, 128, 128), (15, 1), 2_211_840, 15_482_880),
    ),
)
def test_lite_layout_is_byte_exact_at_tp8_and_tp1(
    tp_size,
    conv_shape,
    recurrent_shape,
    packing,
    plane_bytes,
    parent_bytes,
) -> None:
    recipe, groups, layout = _layout(tp_size)
    specs = {spec.group_id: spec for spec, _ in groups}
    fields = {field.field_id: field for field in layout.fields}

    assert Counter(recipe.target_group_ids) == {
        FULL_ATTENTION: 7,
        **{group_id: 7 for group_id in _STATE_GROUPS},
    }
    assert tuple(specs) == (
        _STATE_GROUPS[0],
        FULL_ATTENTION,
        *_STATE_GROUPS[1:],
        "lite_oe",
    )
    assert specs[FULL_ATTENTION].transfer_policy == "full_suffix"
    assert all(
        specs[group_id].transfer_policy == "latest_snapshot"
        for group_id in _STATE_GROUPS
    )
    assert specs["lite_oe"].transfer_policy == "latest_snapshot"
    assert fields["layer.0.conv_state"].shape == conv_shape
    assert fields["layer.0.conv_state"].dtype == "bfloat16"
    assert fields["layer.0.recurrent_state"].shape == recurrent_shape
    assert fields["layer.0.recurrent_state"].dtype == "float32"
    assert fields["layer.3.latent_kv"].shape == (128, 1, 576)
    assert fields["layer.3.latent_kv"].payload_bytes == 147_456
    group_packing = dict(layout.group_packing)
    assert group_packing[FULL_ATTENTION] == packing[0]
    assert all(group_packing[group_id] == packing[1] for group_id in _STATE_GROUPS)
    assert group_packing["lite_oe"] == plane_bytes // 12
    assert len(layout.plane_bytes) == 7
    assert {size for _, size in layout.plane_bytes} == {plane_bytes}
    assert layout.lcm_block_bytes == parent_bytes


def test_lite_oe_does_not_move_existing_kimi_fields() -> None:
    try:
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.kimi_k3 import (
            KimiK3Recipe,
        )
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack
    except ModuleNotFoundError as exc:
        if exc.name == "compressed_tensors":
            pytest.skip("full attention config dependencies are not installed")
        raise

    recipe, _, layout = _layout(8)
    baseline = KimiK3Recipe(
        server_args=recipe.server_args,
        model_config=recipe.model_config,
        attn_config=recipe.attn_config,
        draft_model_config=recipe.draft_model_config,
        draft_attn_config=recipe.draft_attn_config,
        cache_budget_bytes=recipe.cache_budget_bytes,
        decode_input_tokens=recipe.decode_input_tokens,
        overlap_schedule_depth=recipe.overlap_schedule_depth,
    )
    baseline_groups = baseline.groups()
    baseline_layout = pack(
        baseline_groups,
        prefix_granularity=baseline.prefix_granularity,
        cache_blocks_per_lcm_block=baseline.packing(baseline_groups),
        alignment=baseline.alignment,
        max_padding_fraction=baseline.max_padding_fraction,
    )

    assert (
        tuple(field for field in layout.fields if field.group_id != "lite_oe")
        == baseline_layout.fields
    )
    assert (
        tuple(item for item in layout.group_packing if item[0] != "lite_oe")
        == baseline_layout.group_packing
    )


def test_lite_tp8_capacity_accounts_for_history_and_request_state() -> None:
    token_limit = 2 * 1024 * 1024
    recipe, _, layout = _layout(
        8,
        max_bs=256,
        token_limit=token_limit,
        overlap_schedule_depth=1,
    )

    assert recipe.parents_needed(layout, token_limit) == 9985
    assert recipe.token_capacity(layout, 9985) == token_limit


def test_lite_pd_manifest_restores_only_the_latest_oe_context() -> None:
    from test.runtime.test_lite_model_loader import lite_config_dict

    from tokenspeed.runtime.models.lite import (
        LiteNgramParameters,
        LiteOEStatePreparer,
    )
    from tokenspeed.runtime.pd.cache_protocol import (
        CacheTransferContract,
        build_cache_block_manifest,
    )

    _, groups, unbound = _layout(8)
    plan = unbound.bind(2)
    contract = CacheTransferContract(
        plan=plan,
        group_specs=tuple(spec for spec, _ in groups),
    )
    tables = {
        spec.group_id: torch.tensor([[1, 2]], dtype=torch.int32) for spec, _ in groups
    }
    operation = SimpleNamespace(block_tables_arrays=lambda: tables)
    manifest = build_cache_block_manifest(
        operation,
        layout=contract,
        request_row=0,
        prefix_len=0,
        prompt_len=129,
    )
    oe_blocks = next(group for group in manifest.groups if group.group_id == "lite_oe")
    assert oe_blocks.block_ids == (2,)
    assert plan.field("layer.0.lite.oe.context").payload_bytes == 12

    config = LiteConfig.from_dict(lite_config_dict())
    layer = LiteNgramParameters(config)
    for table_id, table in enumerate(layer.embedders):
        table.weight.data = torch.arange(
            config.oe_table_rows(table_id) * config.oe_hidden_size,
            dtype=torch.bfloat16,
        ).reshape(config.oe_table_rows(table_id), config.oe_hidden_size)
    source_pages = torch.zeros((3, 3), dtype=torch.int32)
    destination_pages = torch.zeros_like(source_pages)
    prefill = LiteOEStatePreparer(
        layer,
        context_pages=source_pages,
        checkpoint_granularity=128,
        max_request_slots=1,
        max_graph_tokens=1,
        device="cpu",
    )
    prompt = torch.arange(5, 133).remainder(config.vocab_size)
    prefill.prepare(
        request_ids=["request"],
        request_pool_indices=[0],
        input_ids=prompt,
        lengths=[128],
        before_lengths=[0],
        block_table=tables["lite_oe"],
    )
    prefill.prepare(
        request_ids=["request"],
        request_pool_indices=[0],
        input_ids=torch.tensor([13]),
        lengths=[1],
        before_lengths=[128],
        block_table=tables["lite_oe"],
    )
    destination_pages[oe_blocks.block_ids[0]].copy_(
        source_pages[oe_blocks.block_ids[0]]
    )

    decode = LiteOEStatePreparer(
        layer,
        context_pages=destination_pages,
        checkpoint_granularity=128,
        max_request_slots=1,
        max_graph_tokens=1,
        device="cpu",
    )
    next_token = torch.tensor([17])
    uninterrupted = prefill.prepare(
        request_ids=["request"],
        request_pool_indices=[0],
        input_ids=next_token,
        lengths=[1],
        before_lengths=[129],
        block_table=tables["lite_oe"],
    )
    restored = decode.prepare(
        request_ids=["request"],
        request_pool_indices=[0],
        input_ids=next_token,
        lengths=[1],
        before_lengths=[129],
        block_table=tables["lite_oe"],
    )

    assert decode.restore_count == 1
    assert torch.equal(restored, uninterrupted)
    assert torch.equal(destination_pages[2], source_pages[2])


def test_lite_setup_dispatches_to_the_oe_recipe() -> None:
    try:
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
            prepare_cache_setup,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "compressed_tensors":
            pytest.skip("full attention config dependencies are not installed")
        raise

    recipe = _lite_recipe(8)
    setup = prepare_cache_setup(
        family="kimi_k3",
        server_args=recipe.server_args,
        model_config=recipe.model_config,
        attn_config=recipe.attn_config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=recipe.cache_budget_bytes,
        decode_input_tokens=recipe.decode_input_tokens,
        overlap_schedule_depth=recipe.overlap_schedule_depth,
    )

    assert setup.spec.memory_plan.field("layer.0.lite.oe.context").shape == (3,)


def test_lite_pool_binds_one_arena_and_distinct_layer_views() -> None:
    _, pool = _pool()
    groups = Counter(pool.state_group_by_layer.values())
    conv0, recurrent0 = pool.get_state_buffers(0)
    conv1, recurrent1 = pool.get_state_buffers(1)

    assert pool.requires_page_zeroing
    assert pool.arena.buffer.numel() == pool.arena.plan.arena_bytes
    assert groups == {group_id: 7 for group_id in _STATE_GROUPS}
    assert conv0.shape == (9, 1536, 3)
    assert recurrent0.shape == (9, 4, 128, 128)
    assert conv0.untyped_storage().data_ptr() == pool.arena.buffer.data_ptr()
    assert conv0.data_ptr() != conv1.data_ptr()
    assert recurrent0.data_ptr() != recurrent1.data_ptr()
    assert not torch.count_nonzero(pool.arena.buffer)


def test_lite_graph_state_indices_refresh_without_reallocation() -> None:
    try:
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
        from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
            MambaAttnBackend,
        )
    except RuntimeError as exc:
        if "requires an NVIDIA CUDA, AMD ROCm, or Ascend NPU device" in str(exc):
            pytest.skip("graph metadata runtime requires an accelerator platform")
        raise

    recipe, pool = _pool()
    backend = MambaAttnBackend(recipe.attn_config, recipe.attn_config.components[0])
    backend.set_kv_pool(pool)
    backend.init_cuda_graph_state(2)
    backend.init_forward_metadata_capture_cuda_graph(
        bs=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
    )
    pointers = {
        group_id: (
            backend.state_in_by_group[group_id][1].data_ptr(),
            backend.state_out_by_group[group_id][1].data_ptr(),
        )
        for group_id in _STATE_GROUPS
    }
    tables = {
        _STATE_GROUPS[0]: torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        _STATE_GROUPS[1]: torch.tensor([[5, 6], [7, 8]], dtype=torch.int32),
        _STATE_GROUPS[2]: torch.tensor([[2, 4], [6, 8]], dtype=torch.int32),
    }
    forward_op = object()
    metadata = _OperationMetadata(tables, forward_op)

    backend.init_forward_metadata_replay_cuda_graph(
        bs=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([129, 1], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        num_padding=1,
        cache_metadata=metadata,
        forward_batch=forward_op,
    )

    for group_id in _STATE_GROUPS:
        state_in = backend.forward_metadata.state_in_blocks_by_group[group_id]
        state_out = backend.forward_metadata.state_out_blocks_by_group[group_id]
        assert (state_in.data_ptr(), state_out.data_ptr()) == pointers[group_id]
        assert state_in.tolist() == [int(tables[group_id][0, 0]), -1]
        assert state_out.tolist() == [int(tables[group_id][0, 1]), -1]


def test_lite_reused_state_block_is_zeroed_on_accelerator() -> None:
    device = _accelerator_device()
    _, pool = _pool(device)
    conv, recurrent = pool.get_state_buffers(0)
    conv[0].fill_(5)
    recurrent[0].fill_(5)
    conv[1].fill_(7)
    recurrent[1].fill_(7)
    conv[2].fill_(9)
    recurrent[2].fill_(9)

    pool.zero_new_blocks({_STATE_GROUPS[0]: [1]})
    getattr(torch, device).synchronize()

    assert not torch.count_nonzero(conv[1]).item()
    assert not torch.count_nonzero(recurrent[1]).item()
    assert torch.all(conv[0] == 5).item()
    assert torch.all(recurrent[0] == 5).item()
    assert torch.all(conv[2] == 9).item()
    assert torch.all(recurrent[2] == 9).item()
