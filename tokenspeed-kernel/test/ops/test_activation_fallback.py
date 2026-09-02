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
from tokenspeed_kernel.ops.activation import sigmoid_mul


def test_sigmoid_mul_cpu_fallback_preserves_inplace_and_strided_gate() -> None:
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    packed = torch.randn(3, 2, 8, dtype=torch.bfloat16)
    gate = packed[..., 4:]
    expected = x.float() * torch.sigmoid(gate.reshape_as(x).float())
    pointer = x.data_ptr()

    output = sigmoid_mul(x, gate)

    assert output.data_ptr() == pointer
    torch.testing.assert_close(output.float(), expected, rtol=0.02, atol=0.02)


def test_sigmoid_mul_cpu_fallback_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        sigmoid_mul(torch.ones(2, 8), torch.ones(2, 7))
