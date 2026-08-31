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

from tokenspeed.runtime.engine.request_prefix_inputs import RequestPrefixInputBuilder


class _ForwardOp(SimpleNamespace):
    def num_extends(self) -> int:
        return len(self.extend_prefix_lens)


def test_request_prefix_inputs_follow_boundaries_and_slot_ownership():
    builder = RequestPrefixInputBuilder(lookback=3)
    states = {
        "request-a": SimpleNamespace(
            prompt_input_ids=[10, 11, 12, 13],
            output_ids=[20, 21],
        ),
        "request-b": SimpleNamespace(prompt_input_ids=[30, 31], output_ids=[]),
    }

    extend = _ForwardOp(
        request_ids=["request-a"],
        request_pool_indices=[1],
        extend_prefix_lens=[5],
    )
    assert builder.gather(extend, states) == ((1, 5, (12, 13, 20)),)

    decode = _ForwardOp(
        request_ids=["request-a"],
        request_pool_indices=[1],
        extend_prefix_lens=[],
    )
    assert builder.gather(decode, states) is None

    reused_slot = _ForwardOp(
        request_ids=["request-b"],
        request_pool_indices=[1],
        extend_prefix_lens=[],
    )
    assert builder.gather(reused_slot, states) == ((1, 1, (30,)),)
