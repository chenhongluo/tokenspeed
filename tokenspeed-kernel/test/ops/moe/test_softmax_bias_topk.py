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
from tokenspeed_kernel.ops.moe.softmax_topk import moe_softmax_bias_topk


def test_selection_bias_does_not_enter_route_weights():
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]], dtype=torch.float32)
    bias = torch.tensor([0.0, 0.0, 10.0, 9.0], dtype=torch.float32)
    weights, ids = moe_softmax_bias_topk(
        logits,
        bias,
        2,
        routed_scaling_factor=6.0,
        solution="torch",
    )

    probabilities = logits.softmax(-1)
    expected_ids = torch.tensor([[2, 3]], dtype=torch.int32)
    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(
        weights,
        probabilities.gather(1, expected_ids.long()) * 6.0,
    )
    assert weights.sum() != pytest.approx(6.0)


def test_empty_batch_and_validation():
    logits = torch.empty((0, 4), dtype=torch.bfloat16)
    bias = torch.zeros(4, dtype=torch.bfloat16)
    weights, ids = moe_softmax_bias_topk(logits, bias, 2)
    assert weights.shape == ids.shape == (0, 2)
    assert weights.dtype == torch.float32 and ids.dtype == torch.int32

    with pytest.raises(ValueError, match=r"shape \[experts\]"):
        moe_softmax_bias_topk(torch.zeros(1, 4), torch.zeros(3), 2)
    with pytest.raises(ValueError, match="share a dtype"):
        moe_softmax_bias_topk(
            torch.zeros(1, 4, dtype=torch.float32),
            torch.zeros(4, dtype=torch.bfloat16),
            2,
        )
