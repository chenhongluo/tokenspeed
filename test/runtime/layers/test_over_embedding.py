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
from tokenspeed_kernel.ops.over_embedding import OverEmbeddingSpec, TableFragmentSpec

import tokenspeed.runtime.layers.over_embedding as over_embedding
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


def test_forward_owns_history_publication_and_masks_graph_padding(monkeypatch) -> None:
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
    req_pool_indices = torch.tensor([0, 2], dtype=torch.int64)
    input_lengths = torch.tensor([2, 99], dtype=torch.int32)
    layer.bind_runtime_inputs(
        req_pool_indices=req_pool_indices,
        input_lengths=input_lengths,
        max_request_slots=2,
        history_capacity=16,
        padding_req_pool_index=2,
    )
    layer.stage_prefixes(
        req_pool_indices=(0,),
        prefix_lengths=(3,),
        request_token_ids=((6, 7),),
    )

    assert not torch.count_nonzero(layer.history_token_ids)
    assert layer.committed_lengths.tolist() == [0, 0, 0]

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
        assert committed_lengths[0].item() == 3
        assert history_token_ids[0, 1:3].tolist() == [6, 7]
        history_token_ids[0, 3:5].copy_(input_ids[:2])
        return torch.zeros((3, 6), dtype=torch.bfloat16)

    monkeypatch.setattr(layer.word_embedding, "forward", fake_word)
    monkeypatch.setattr(over_embedding, "append_packed_lookup_", fake_lookup)
    monkeypatch.setattr(
        over_embedding,
        "project_add_word_",
        lambda word_partial, *_args, **_kwargs: word_partial,
    )

    output = layer(
        torch.tensor([8, 9, 1], dtype=torch.int32),
        torch.tensor([3, 4, 0], dtype=torch.int64),
        SimpleNamespace(bs=2),
    )

    assert output.shape == (3, 8)
    assert layer.history_token_ids[0, 1:5].tolist() == [6, 7, 8, 9]
    assert layer.pending_prefix_lengths.tolist() == [-1, -1, -1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_history_publication_is_cuda_graph_capturable(monkeypatch) -> None:
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
    layer.bind_runtime_inputs(
        req_pool_indices=torch.tensor([0], device="cuda", dtype=torch.int64),
        input_lengths=torch.tensor([1], device="cuda", dtype=torch.int32),
        max_request_slots=1,
        history_capacity=8,
        padding_req_pool_index=1,
    )
    layer.stage_prefixes(
        req_pool_indices=(0,),
        prefix_lengths=(0,),
        request_token_ids=((),),
    )
    input_ids = torch.tensor([8], device="cuda", dtype=torch.int32)
    positions = torch.tensor([0], device="cuda", dtype=torch.int64)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layer._prepare_history(input_ids, positions, batch_size=1)
    graph.replay()
    torch.cuda.synchronize()

    assert layer.pending_prefix_lengths[0].item() == -1
    assert layer.pending_prefix_counts[0].item() == 0
