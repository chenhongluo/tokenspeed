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

import torch

from tokenspeed.runtime.execution.input_buffer import InputBuffers


def test_request_token_history_layout_handles_mixed_rows() -> None:
    buffers = InputBuffers.__new__(InputBuffers)
    buffers.max_bs = 3
    buffers.input_lengths_buf = torch.tensor([2, 1, 1], dtype=torch.int32)
    buffers.request_token_history_input_lengths_buf = torch.empty(3, dtype=torch.int32)
    buffers.input_start_offsets_buf = torch.empty(4, dtype=torch.int32)
    buffers.active_request_mask_buf = torch.empty(3, dtype=torch.bool)

    buffers.prepare_request_token_history_inputs(
        batch_size=3,
        num_extends=1,
        decode_width=8,
    )

    assert buffers.input_start_offsets_buf.tolist() == [0, 2, 10, 18]
    assert buffers.active_request_mask_buf.tolist() == [True, True, True]
