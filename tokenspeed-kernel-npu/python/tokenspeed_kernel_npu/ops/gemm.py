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

import torch
import torch_npu


def prepare_weight_nz(weight: torch.Tensor, *, transpose: bool = False) -> torch.Tensor:
    if weight.device.type != "npu" or weight.ndim != 2:
        raise ValueError("Weight-NZ preparation requires a two-dimensional NPU tensor.")
    if transpose:
        weight = weight.transpose(0, 1).contiguous()

    config = torch.npu.config
    previous = getattr(config, "allow_internal_format", None)
    try:
        config.allow_internal_format = True
        prepared = torch_npu.npu_format_cast(weight, 29)
        actual_format = torch_npu.get_npu_format(prepared)
        if actual_format != 29:
            raise RuntimeError(
                f"Weight-NZ preparation requested format 29, got {actual_format}."
            )
        return prepared
    finally:
        config.allow_internal_format = previous
