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

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.over_embedding import (
    OverEmbeddingSpec,
    TableFragmentSpec,
    project_add_word_,
)

import tokenspeed.runtime.layers.over_embedding as over_embedding
from tokenspeed.runtime.execution.request_token_history import RequestTokenHistoryView
from tokenspeed.runtime.layers.over_embedding import (
    LongCatOverEmbedding,
    resolve_longcat_oe_spec,
)


def _small_spec() -> OverEmbeddingSpec:
    return OverEmbeddingSpec(
        profile="unit-test",
        tp_size=1,
        rank=0,
        vocab_size=31,
        branch_count=2,
        branch_width=4,
        hidden_size=8,
        max_ngram_order=3,
        fragments=(
            TableFragmentSpec(0, 2, 7, 0, 4),
            TableFragmentSpec(1, 3, 9, 2, 2),
        ),
    )


def _history_view(
    *,
    history_token_ids: torch.Tensor,
    committed_lengths: torch.Tensor,
    req_pool_indices: torch.Tensor,
    input_start_offsets: torch.Tensor,
    active_request_mask: torch.Tensor,
) -> RequestTokenHistoryView:
    return RequestTokenHistoryView(
        history_token_ids=history_token_ids,
        committed_lengths=committed_lengths,
        req_pool_indices=req_pool_indices,
        input_start_offsets=input_start_offsets,
        active_request_mask=active_request_mask,
    )


def test_flash_lite_profile_matches_checkpoint_geometry() -> None:
    spec = resolve_longcat_oe_spec(
        vocab_size=163840,
        hidden_size=3072,
        max_ngram_order=4,
        hashes_per_order=4,
        modulus0=4718593,
        tp_size=8,
        tp_rank=0,
    )

    assert spec.branch_count == 12
    assert spec.local_width == 384
    assert [fragment.modulus for fragment in spec.fragments] == [4718593, 4718609]


def test_projection_preserves_word_embedding_for_bypassed_tokens() -> None:
    word = torch.tensor([[3.0, 6.0], [5.0, 7.0]], dtype=torch.bfloat16)
    activation = torch.tensor([[2.0], [0.0]], dtype=torch.bfloat16)
    projection = torch.tensor([[9.0, 12.0]], dtype=torch.bfloat16)

    output = project_add_word_(
        word,
        activation,
        projection,
        scale=3.0,
        bypass_mask=torch.tensor([False, True]),
        solution="torch",
    )

    torch.testing.assert_close(
        output[0], torch.tensor([7.0, 10.0], dtype=torch.bfloat16)
    )
    torch.testing.assert_close(
        output[1], torch.tensor([5.0, 7.0], dtype=torch.bfloat16)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_forward_bypass_mask_supports_cuda_graph_replay(monkeypatch) -> None:
    monkeypatch.setattr(
        over_embedding, "resolve_longcat_oe_spec", lambda **_: _small_spec()
    )
    layer = LongCatOverEmbedding(
        num_embeddings=31,
        embedding_dim=8,
        over_embedding_m=6,
        hashes_per_order=1,
        max_ngram_order=3,
        tp_rank=0,
        tp_size=1,
        tp_group=(0,),
        params_dtype=torch.bfloat16,
        ignored_token_ids=tuple(range(23)),
        segment_ignored_tokens=True,
    ).cuda()
    history_view = _history_view(
        history_token_ids=torch.zeros((1, 16), dtype=torch.int32, device="cuda"),
        committed_lengths=torch.zeros(1, dtype=torch.int32, device="cuda"),
        req_pool_indices=torch.zeros(1, dtype=torch.int64, device="cuda"),
        input_start_offsets=torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
        active_request_mask=torch.ones(1, dtype=torch.bool, device="cuda"),
    )

    def fake_lookup(input_ids, *_args, **_kwargs):
        return torch.zeros(
            (input_ids.numel(), layer.spec.local_width),
            dtype=torch.bfloat16,
            device=input_ids.device,
        )

    def return_bypass_mask(
        _word_partial,
        _activation,
        _projection,
        *,
        scale,
        bypass_mask,
    ):
        assert scale == layer.normalize_scale
        return bypass_mask

    monkeypatch.setattr(over_embedding, "append_packed_lookup_", fake_lookup)
    monkeypatch.setattr(over_embedding, "project_add_word_", return_bypass_mask)

    input_ids = torch.tensor([30, 22], dtype=torch.int32, device="cuda")
    ctx = SimpleNamespace(request_token_history=history_view)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        layer(input_ids, ctx)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = layer(input_ids, ctx)

    input_ids.copy_(torch.tensor([21, 29], dtype=torch.int32, device="cuda"))
    graph.replay()

    assert output.cpu().tolist() == [True, False]


def test_forward_uses_runtime_history_and_masks_graph_padding(monkeypatch) -> None:
    monkeypatch.setattr(
        over_embedding, "resolve_longcat_oe_spec", lambda **_: _small_spec()
    )
    layer = LongCatOverEmbedding(
        num_embeddings=31,
        embedding_dim=8,
        over_embedding_m=6,
        hashes_per_order=1,
        max_ngram_order=3,
        tp_rank=0,
        tp_size=1,
        tp_group=(0,),
        params_dtype=torch.bfloat16,
        fix_normalize_factor=True,
    )
    assert layer.normalize_scale == pytest.approx(3**0.5)
    history_token_ids = torch.zeros((3, 16), dtype=torch.int32)
    committed_lengths = torch.tensor([3, 0, 0], dtype=torch.int32)
    history_token_ids[0, 1:3] = torch.tensor([6, 7], dtype=torch.int32)
    history_view = _history_view(
        history_token_ids=history_token_ids,
        committed_lengths=committed_lengths,
        req_pool_indices=torch.tensor([0, 2], dtype=torch.int64),
        input_start_offsets=torch.tensor([0, 2, 3], dtype=torch.int32),
        active_request_mask=torch.tensor([True, False]),
    )
    assert not hasattr(layer, "_req_pool_indices")
    assert not hasattr(layer, "_input_lengths")
    assert not hasattr(layer, "_history_token_ids")
    assert not hasattr(layer, "_committed_lengths")

    def fake_word(input_ids, *, reduce_results=False):
        assert not reduce_results
        return torch.zeros((input_ids.numel(), 8), dtype=torch.bfloat16)

    def fake_lookup(
        input_ids,
        input_start_offsets,
        req_pool_indices,
        active_request_mask,
        history_token_ids,
        committed_lengths,
        oe_tables,
        **_,
    ):
        assert input_start_offsets.tolist() == [0, 2, 3]
        assert req_pool_indices.tolist() == [0, 2]
        assert active_request_mask.tolist() == [True, False]
        assert history_token_ids is history_token_ids_runtime
        assert committed_lengths is committed_lengths_runtime
        assert committed_lengths[0].item() == 3
        assert history_token_ids[0, 1:3].tolist() == [6, 7]
        history_token_ids[0, 3:5].copy_(input_ids[:2])
        return torch.zeros((3, 6), dtype=torch.bfloat16)

    history_token_ids_runtime = history_token_ids
    committed_lengths_runtime = committed_lengths

    monkeypatch.setattr(layer.word_embedding, "forward", fake_word)
    monkeypatch.setattr(over_embedding, "append_packed_lookup_", fake_lookup)
    monkeypatch.setattr(
        over_embedding,
        "project_add_word_",
        lambda word_partial, *_args, **_kwargs: word_partial,
    )

    output = layer(
        torch.tensor([8, 9, 1], dtype=torch.int32),
        SimpleNamespace(request_token_history=history_view),
    )

    assert output.shape == (3, 8)
    assert history_token_ids[0, 1:5].tolist() == [6, 7, 8, 9]


def test_forward_splits_dflash_verify_rows_before_graph_padding(monkeypatch) -> None:
    monkeypatch.setattr(
        over_embedding, "resolve_longcat_oe_spec", lambda **_: _small_spec()
    )
    layer = LongCatOverEmbedding(
        num_embeddings=31,
        embedding_dim=8,
        over_embedding_m=6,
        hashes_per_order=1,
        max_ngram_order=3,
        tp_rank=0,
        tp_size=1,
        tp_group=(0,),
        params_dtype=torch.bfloat16,
    )
    history_token_ids = torch.zeros((3, 32), dtype=torch.int32)
    committed_lengths = torch.tensor([10, 20, 0], dtype=torch.int32)
    history_view = _history_view(
        history_token_ids=history_token_ids,
        committed_lengths=committed_lengths,
        req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
        input_start_offsets=torch.tensor([0, 8, 16, 24], dtype=torch.int32),
        active_request_mask=torch.tensor([True, True, False]),
    )

    def fake_word(input_ids, *, reduce_results=False):
        assert not reduce_results
        return torch.zeros((input_ids.numel(), 8), dtype=torch.bfloat16)

    def fake_lookup(
        input_ids,
        input_start_offsets,
        req_pool_indices,
        active_request_mask,
        history_token_ids,
        committed_lengths,
        oe_tables,
        **_,
    ):
        assert input_start_offsets.tolist() == [0, 8, 16, 24]
        assert req_pool_indices.tolist() == [0, 1, 2]
        assert active_request_mask.tolist() == [True, True, False]
        assert committed_lengths.tolist() == [10, 20, 0]
        return torch.zeros((input_ids.numel(), 6), dtype=torch.bfloat16)

    monkeypatch.setattr(layer.word_embedding, "forward", fake_word)
    monkeypatch.setattr(over_embedding, "append_packed_lookup_", fake_lookup)
    monkeypatch.setattr(
        over_embedding,
        "project_add_word_",
        lambda word_partial, *_args, **_kwargs: word_partial,
    )

    output = layer(
        torch.arange(24, dtype=torch.int32),
        SimpleNamespace(request_token_history=history_view),
    )

    assert output.shape == (24, 8)
