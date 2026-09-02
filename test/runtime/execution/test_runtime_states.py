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

import pytest
import torch

from tokenspeed.runtime.execution.runtime_states import RuntimeStates


def _runtime_states() -> RuntimeStates:
    return RuntimeStates(
        req_pool_size=2,
        vocab_size=32,
        output_length=1,
        request_token_history_capacity=8,
        device="cpu",
    )


def test_runtime_states_owns_and_seeds_request_token_history() -> None:
    states = _runtime_states()
    states.reset_states(
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([5], dtype=torch.int32),
    )

    states.seed_request_token_history(
        req_pool_indices=(1,),
        prefix_lengths=(5,),
        request_token_ids=((4, 5, 6, 7, 8),),
    )

    assert states.request_token_history_ids is not None
    assert states.request_token_history_ids[1, :5].tolist() == [4, 5, 6, 7, 8]
    assert states.valid_cache_lengths[1].item() == 5


def test_runtime_states_builds_a_non_owning_batch_view() -> None:
    states = _runtime_states()
    req_pool_indices = torch.tensor([1, 2], dtype=torch.int64)
    input_start_offsets = torch.tensor([0, 2, 3], dtype=torch.int32)
    active_request_mask = torch.tensor([True, False])

    view = states.request_token_history_view(
        req_pool_indices=req_pool_indices,
        input_start_offsets=input_start_offsets,
        active_request_mask=active_request_mask,
    )

    assert view is not None
    assert view.history_token_ids is states.request_token_history_ids
    assert view.committed_lengths is states.valid_cache_lengths
    assert view.req_pool_indices is req_pool_indices
    assert view.input_start_offsets is input_start_offsets
    assert view.active_request_mask is active_request_mask


def test_request_token_history_rejects_invalid_prefix_inputs() -> None:
    states = _runtime_states()

    with pytest.raises(ValueError, match="matching lengths"):
        states.seed_request_token_history(
            req_pool_indices=(0, 1),
            prefix_lengths=(0,),
            request_token_ids=((),),
        )
    with pytest.raises(ValueError, match="slot 2 is out of range"):
        states.seed_request_token_history(
            req_pool_indices=(2,),
            prefix_lengths=(0,),
            request_token_ids=((),),
        )
    with pytest.raises(ValueError, match="must cover prefix length 4"):
        states.seed_request_token_history(
            req_pool_indices=(0,),
            prefix_lengths=(4,),
            request_token_ids=((1, 2, 3),),
        )


def test_request_token_history_must_be_enabled_before_seeding() -> None:
    states = RuntimeStates(
        req_pool_size=2,
        vocab_size=32,
        output_length=1,
        device="cpu",
    )
    assert states.request_token_history_ids is None

    with pytest.raises(RuntimeError, match="not enabled"):
        states.seed_request_token_history(
            req_pool_indices=(0,),
            prefix_lengths=(0,),
            request_token_ids=((),),
        )
