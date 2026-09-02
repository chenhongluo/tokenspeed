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

from tokenspeed.runtime.engine.event_loop import EventLoop


class _ForwardOp(SimpleNamespace):
    def num_extends(self) -> int:
        return len(self.extend_prefix_lens)


def _event_loop(states):
    loop = EventLoop.__new__(EventLoop)
    loop._requires_request_token_history = True
    loop.output_processor = SimpleNamespace(rid_to_state=states)
    return loop


def test_request_history_seeds_copy_full_extend_prefixes():
    states = {
        "request-a": SimpleNamespace(
            prompt_input_ids=[10, 11, 12, 13],
            output_ids=[20, 21],
        ),
        "request-b": SimpleNamespace(prompt_input_ids=[30, 31], output_ids=[]),
    }

    mixed = _ForwardOp(
        request_ids=["request-a", "request-b"],
        request_pool_indices=[1, 2],
        extend_prefix_lens=[5],
    )
    assert _event_loop(states)._gather_request_history_seeds(mixed) == (
        (1, 5, (10, 11, 12, 13, 20)),
    )


def test_request_history_seeds_skip_empty_boundaries_and_decode_batches():
    states = {}
    empty_boundary = _ForwardOp(
        request_ids=["request-a"],
        request_pool_indices=[1],
        extend_prefix_lens=[0],
    )
    loop = _event_loop(states)
    assert loop._gather_request_history_seeds(empty_boundary) is None

    decode = _ForwardOp(
        request_ids=["request-a"],
        request_pool_indices=[1],
        extend_prefix_lens=[],
    )
    assert loop._gather_request_history_seeds(decode) is None


def test_request_history_seeds_skip_models_without_request_history():
    loop = _event_loop({})
    loop._requires_request_token_history = False
    extend = _ForwardOp(
        request_ids=["request-a"],
        request_pool_indices=[1],
        extend_prefix_lens=[1],
    )

    assert loop._gather_request_history_seeds(extend) is None
