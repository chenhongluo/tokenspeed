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

import gc
from test.runtime.test_lite_model_loader import (
    lite_config_dict,
    mapping,
    weights,
)
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from tokenspeed.runtime.configs.lite_config import LiteConfig
from tokenspeed.runtime.models.lite import (
    FLASHLocalForCausalLM,
    LiteNgramParameters,
    LiteOEStatePreparer,
)


def _brute_ids(config, flat_tokens, initial_context, lengths):
    outputs = []
    tails = []
    special = set(config.special_token_ids)
    offset = 0
    for request_id, length in enumerate(lengths):
        history = initial_context[request_id].tolist()
        for token in flat_tokens[offset : offset + length].tolist():
            window = history[-3:] + [token]
            clean = [0 if value in special else value for value in window]
            row = []
            for order in range(2, config.emb_neighbor_num + 1):
                for split_id in range(config.emb_split_num):
                    table_id = (order - 2) * config.emb_split_num + split_id
                    rows = config.oe_table_rows(table_id)
                    value = clean[-1]
                    for distance in range(1, order):
                        if all(clean[-1 - distance :]):
                            value += clean[-1 - distance] * pow(
                                config.vocab_size, distance, rows
                            )
                    row.append(value % rows)
            outputs.append(row)
            history.append(token)
        tails.append(history[-3:])
        offset += length
    return (
        torch.tensor(outputs, dtype=torch.int64).reshape(-1, 12),
        torch.tensor(tails, dtype=torch.int64),
    )


def test_lite_oe_ids_match_hand_computation_and_chunk_continuation() -> None:
    config = LiteConfig.from_dict(lite_config_dict())
    layer = LiteNgramParameters(config)
    initial = torch.tensor([[2, 2, 5], [2, 2, 2], [7, 8, 9]])
    tokens = torch.tensor([6, 36, 10, 11, 12, 2, 13, 14], dtype=torch.int64)
    lengths = [3, 5, 0]

    ids, special, tail = layer.ngram_ids(tokens, initial, lengths)
    expected_ids, expected_tail = _brute_ids(config, tokens, initial, lengths)

    assert torch.equal(ids, expected_ids)
    assert torch.equal(tail, expected_tail)
    assert torch.equal(
        special,
        torch.tensor([False, True, False, False, False, True, False, False]),
    )

    one_shot_ids, _, one_shot_tail = layer.ngram_ids(
        torch.tensor([15, 16, 17, 18]), initial[:1], [4]
    )
    first_ids, _, first_tail = layer.ngram_ids(torch.tensor([15, 16]), initial[:1], [2])
    second_ids, _, second_tail = layer.ngram_ids(
        torch.tensor([17, 18]), first_tail, [2]
    )
    assert torch.equal(torch.cat([first_ids, second_ids]), one_shot_ids)
    assert torch.equal(second_tail, one_shot_tail)


def test_lite_oe_lookup_projection_normalize_and_special() -> None:
    torch.manual_seed(19)
    config = LiteConfig.from_dict(lite_config_dict())
    layer = LiteNgramParameters(config)
    for table_id, table in enumerate(layer.embedders):
        rows = config.oe_table_rows(table_id)
        table.weight.data = (
            torch.arange(rows * config.oe_hidden_size, dtype=torch.float32)
            .reshape(rows, config.oe_hidden_size)
            .remainder(31)
            .to(torch.bfloat16)
        )
    layer.projection.data.uniform_(-0.02, 0.02)
    ids = torch.stack(
        [
            torch.arange(3).remainder(config.oe_table_rows(table_id))
            for table_id in range(config.oe_component_count)
        ],
        dim=1,
    ).to(torch.int64)

    raw = layer.lookup_host(ids)
    expected_raw = torch.stack(
        [layer.embedders[i].weight[ids[:, i]] for i in range(12)], dim=1
    )
    assert torch.equal(raw, expected_raw)

    word = torch.randn(3, config.hidden_size, dtype=torch.bfloat16)
    input_ids = torch.tensor([5, 36, 8])
    actual = layer.project_and_merge(word, raw, input_ids)
    projected = sum(raw[:, i].float() @ layer.projection[i].float() for i in range(12))
    expected = (word.float() + projected) / (13**0.5)
    expected[1] = word[1].float()

    assert torch.equal(actual[1], word[1])
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual.float(), expected, atol=0.02, rtol=0.02)
    empty = layer.project_and_merge(
        word[:0], raw[:0], torch.empty(0, dtype=torch.int64)
    )
    assert empty.shape == (0, config.hidden_size)


def test_lite_oe_state_restore_slot_reuse_and_fixed_staging() -> None:
    config = LiteConfig.from_dict(lite_config_dict())
    layer = LiteNgramParameters(config)
    for table_id, table in enumerate(layer.embedders):
        rows = config.oe_table_rows(table_id)
        table.weight.data = (
            torch.arange(rows * config.oe_hidden_size, dtype=torch.float32)
            .reshape(rows, config.oe_hidden_size)
            .remainder(17)
            .to(torch.bfloat16)
        )
    pages = torch.zeros((8, 3), dtype=torch.int32)
    preparer = LiteOEStatePreparer(
        layer,
        context_pages=pages,
        checkpoint_granularity=128,
        max_request_slots=2,
        max_graph_tokens=4,
        device="cpu",
    )
    block_table = torch.tensor([[1, 2]], dtype=torch.int32)

    first = preparer.prepare(
        request_ids=["r0"],
        request_pool_indices=[0],
        input_ids=torch.tensor([5, 6]),
        lengths=[2],
        before_lengths=[0],
        block_table=block_table,
        graph_tokens=4,
    )
    pointer = first.data_ptr()
    assert pages[1].tolist() == [2, 5, 6]
    assert not torch.count_nonzero(first[2:])
    assert preparer.restore_count == 0

    second = preparer.prepare(
        request_ids=["r0"],
        request_pool_indices=[0],
        input_ids=torch.tensor([7]),
        lengths=[1],
        before_lengths=[2],
        block_table=block_table,
        graph_tokens=4,
    )
    assert second.data_ptr() == pointer
    assert pages[1].tolist() == [5, 6, 7]
    assert not torch.count_nonzero(second[1:])
    assert preparer.restore_count == 0

    pages[2] = torch.tensor([8, 9, 10], dtype=torch.int32)
    reused = preparer.prepare(
        request_ids=["r1"],
        request_pool_indices=[0],
        input_ids=torch.tensor([11]),
        lengths=[1],
        before_lengths=[129],
        block_table=block_table,
        graph_tokens=4,
    )
    assert reused.data_ptr() == pointer
    assert pages[2].tolist() == [9, 10, 11]
    assert preparer.restore_count == 1


def test_lite_oe_state_crosses_checkpoint_boundary_without_restore() -> None:
    config = LiteConfig.from_dict(lite_config_dict())
    layer = LiteNgramParameters(config)
    for table_id, table in enumerate(layer.embedders):
        rows = config.oe_table_rows(table_id)
        table.weight.data = torch.zeros(
            (rows, config.oe_hidden_size), dtype=torch.bfloat16
        )
    pages = torch.zeros((4, 3), dtype=torch.int32)
    preparer = LiteOEStatePreparer(
        layer,
        context_pages=pages,
        checkpoint_granularity=128,
        max_request_slots=1,
        max_graph_tokens=2,
        device="cpu",
    )
    block_table = torch.tensor([[1, 2]], dtype=torch.int32)
    tokens = torch.arange(5, 133).remainder(config.vocab_size)
    preparer.prepare(
        request_ids=["r0"],
        request_pool_indices=[0],
        input_ids=tokens,
        lengths=[128],
        before_lengths=[0],
        block_table=block_table,
    )
    assert pages[1].tolist() == tokens[-3:].tolist()

    preparer.prepare(
        request_ids=["r0"],
        request_pool_indices=[0],
        input_ids=torch.tensor([133 % config.vocab_size]),
        lengths=[1],
        before_lengths=[128],
        block_table=block_table,
        graph_tokens=2,
    )
    assert pages[2].tolist() == [
        int(tokens[-2]),
        int(tokens[-1]),
        133 % config.vocab_size,
    ]
    assert preparer.restore_count == 0


def test_lite_oe_uses_resolved_device_owned_decode_token() -> None:
    config = LiteConfig.from_dict(lite_config_dict())
    layer = LiteNgramParameters(config)
    for table_id, table in enumerate(layer.embedders):
        table.weight.data = torch.zeros(
            (config.oe_table_rows(table_id), config.oe_hidden_size),
            dtype=torch.bfloat16,
        )
    pages = torch.zeros((2, 3), dtype=torch.int32)
    pages[1] = torch.tensor([2, 5, 6], dtype=torch.int32)
    preparer = LiteOEStatePreparer(
        layer,
        context_pages=pages,
        checkpoint_granularity=128,
        max_request_slots=1,
        max_graph_tokens=1,
        device="cpu",
    )
    forward_op = SimpleNamespace(
        request_ids=["r0"],
        request_pool_indices=[0],
        input_lengths=[1],
        input_ids=[],
        decode_input_ids=[-1],
        extend_prefix_lens=[],
        prefill_lengths=[3],
        num_extends=lambda: 0,
        block_tables_arrays=lambda: {"lite_oe": torch.tensor([[1]])},
    )

    raw = preparer.prepare_forward_op(
        forward_op,
        resolved_input_ids=torch.tensor([7], dtype=torch.int32),
        graph_tokens=1,
    )

    assert raw.shape == (1, config.oe_component_count, config.oe_hidden_size)
    assert pages[1].tolist() == [5, 6, 7]


def test_lite_oe_rejects_resolved_token_count_before_publication() -> None:
    config = LiteConfig.from_dict(lite_config_dict())
    pages = torch.zeros((2, 3), dtype=torch.int32)
    preparer = LiteOEStatePreparer(
        LiteNgramParameters(config),
        context_pages=pages,
        checkpoint_granularity=128,
        max_request_slots=1,
        max_graph_tokens=1,
        device="cpu",
    )
    forward_op = SimpleNamespace(
        request_ids=["r0"],
        request_pool_indices=[0],
        input_lengths=[1],
        input_ids=[],
        decode_input_ids=[-1],
        extend_prefix_lens=[],
        prefill_lengths=[3],
        num_extends=lambda: 0,
        block_tables_arrays=lambda: {"lite_oe": torch.tensor([[1]])},
    )

    with pytest.raises(ValueError, match="resolved input IDs"):
        preparer.prepare_forward_op(
            forward_op,
            resolved_input_ids=torch.empty(0, dtype=torch.int32),
            graph_tokens=1,
        )

    assert not torch.count_nonzero(pages)
    assert preparer.restore_count == 0


def test_lite_oe_rejects_invalid_slots_and_pages_before_publication() -> None:
    config = LiteConfig.from_dict(lite_config_dict())
    pages = torch.zeros((2, 3), dtype=torch.int32)
    preparer = LiteOEStatePreparer(
        LiteNgramParameters(config),
        context_pages=pages,
        checkpoint_granularity=128,
        max_request_slots=2,
        max_graph_tokens=2,
        device="cpu",
    )
    common = {
        "request_ids": ["r0"],
        "request_pool_indices": [0],
        "input_ids": torch.tensor([5]),
        "lengths": [1],
    }

    with pytest.raises(ValueError, match="request-pool index"):
        preparer.prepare(
            **{**common, "request_pool_indices": [2]},
            before_lengths=[0],
            block_table=torch.tensor([[1]]),
        )
    with pytest.raises(ValueError, match="page 0"):
        preparer.prepare(
            **common,
            before_lengths=[0],
            block_table=torch.tensor([[0]]),
        )
    with pytest.raises(ValueError, match="out of range"):
        preparer.prepare(
            **common,
            before_lengths=[0],
            block_table=torch.tensor([[2]]),
        )
    with pytest.raises(ValueError, match="multiple owners"):
        preparer.prepare(
            request_ids=["r0", "r1"],
            request_pool_indices=[0, 1],
            input_ids=torch.tensor([5, 6]),
            lengths=[1, 1],
            before_lengths=[0, 0],
            block_table=torch.tensor([[1], [1]]),
        )
    assert not torch.count_nonzero(pages)


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires an Ascend NPU",
)
def test_lite_oe_fixed_staging_replays_updated_values() -> None:
    torch.npu.set_device(0)
    torch.manual_seed(2027)
    config = LiteConfig.from_dict(lite_config_dict())
    layer = LiteNgramParameters(config)
    for table_id, table in enumerate(layer.embedders):
        table.weight.data = torch.randn(
            config.oe_table_rows(table_id),
            config.oe_hidden_size,
            dtype=torch.bfloat16,
        )
    layer.projection.data = torch.randn_like(layer.projection, device="npu") * 0.02
    layer.ignore_tokens = layer.ignore_tokens.to("npu")
    preparer = LiteOEStatePreparer(
        layer,
        context_pages=torch.zeros((4, 3), dtype=torch.int32, device="npu"),
        checkpoint_granularity=128,
        max_request_slots=1,
        max_graph_tokens=2,
        device="npu",
    )
    block_table = torch.tensor([[1]], dtype=torch.int32)
    staging = preparer.prepare(
        request_ids=["r0"],
        request_pool_indices=[0],
        input_ids=torch.tensor([5, 6]),
        lengths=[2],
        before_lengths=[0],
        block_table=block_table,
        graph_tokens=2,
    )
    pointer = staging.data_ptr()
    word = torch.randn(2, config.hidden_size, dtype=torch.bfloat16, device="npu")
    input_ids = torch.tensor([5, 6], dtype=torch.int64, device="npu")

    for _ in range(3):
        layer.project_and_merge(word, staging, input_ids)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        graph_output = layer.project_and_merge(word, staging, input_ids)
    graph.replay()
    torch.npu.synchronize()
    expected = layer.project_and_merge(word, staging, input_ids)
    torch.testing.assert_close(graph_output, expected, atol=0, rtol=0)
    first = graph_output.clone()

    updated = preparer.prepare(
        request_ids=["r0"],
        request_pool_indices=[0],
        input_ids=torch.tensor([7]),
        lengths=[1],
        before_lengths=[2],
        block_table=block_table,
        graph_tokens=2,
    )
    assert updated.data_ptr() == pointer
    assert not torch.count_nonzero(updated[1:]).item()
    word.copy_(torch.randn_like(word.float()).to(torch.bfloat16))
    input_ids.copy_(torch.tensor([7, 1], dtype=torch.int64, device="npu"))
    graph.replay()
    torch.npu.synchronize()
    expected = layer.project_and_merge(word, updated, input_ids)
    torch.testing.assert_close(graph_output, expected, atol=0, rtol=0)
    assert not torch.equal(graph_output, first)
    assert torch.isfinite(graph_output).all()


def test_lite_oe_loader_adopts_safetensors_mapping(tmp_path) -> None:
    config = LiteConfig.from_dict(lite_config_dict())
    model = FLASHLocalForCausalLM(config, mapping())
    name = "model.ngram_embeddings.embedders.0.weight"
    source = torch.arange(13 * 8, dtype=torch.bfloat16).reshape(13, 8)
    checkpoint = tmp_path / "oe.safetensors"
    save_file({name: source}, checkpoint)
    loaded = load_file(checkpoint, device="cpu")
    mapped = loaded[name]
    pointer = mapped.untyped_storage().data_ptr()

    def checkpoint_weights():
        for source_name, tensor in weights(model.layout):
            yield source_name, mapped if source_name == name else tensor

    model.load_weights(checkpoint_weights())
    table = model.model.ngram_embeddings.embedders[0].weight
    assert table.untyped_storage().data_ptr() == pointer

    del mapped, loaded
    gc.collect()
    assert torch.equal(table, source)
    with open("/proc/self/maps") as mappings:
        mapped_line = next(
            line
            for line in mappings
            if int(line.split("-", 1)[0], 16)
            <= pointer
            < int(line.split("-", 1)[1].split()[0], 16)
        )
    assert str(checkpoint) in mapped_line
