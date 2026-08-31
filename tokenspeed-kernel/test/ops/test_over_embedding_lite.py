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
from tokenspeed_kernel.ops.over_embedding import (
    OverEmbeddingSpec,
    TableFragmentSpec,
    append_packed_lookup_,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("exclude_special_tokens", "expected_rows"),
    [(False, [5, 9, 4]), (True, [5, 0, 7])],
)
def test_flash_lite_special_tokens_bypass_and_split_oe_history(
    exclude_special_tokens: bool,
    expected_rows: list[int],
) -> None:
    pytest.importorskip("cutlass.cute")
    spec = OverEmbeddingSpec(
        profile="longcat-lite-test",
        tp_size=1,
        rank=0,
        vocab_size=32,
        branch_count=2,
        branch_width=256,
        hidden_size=384,
        max_ngram_order=3,
        fragments=(
            TableFragmentSpec(0, 2, 11, 0, 256),
            TableFragmentSpec(1, 3, 13, 0, 128),
        ),
        ignored_token_ids=(3,) if exclude_special_tokens else (),
        segment_ignored_tokens=exclude_special_tokens,
    )
    tables = (
        torch.arange(11, device="cuda", dtype=torch.bfloat16)
        .unsqueeze(1)
        .expand(-1, 256)
        .contiguous(),
        torch.arange(13, device="cuda", dtype=torch.bfloat16)
        .unsqueeze(1)
        .expand(-1, 128)
        .contiguous(),
    )
    history = torch.zeros((2, 8), device="cuda", dtype=torch.int32)

    output = append_packed_lookup_(
        torch.tensor([5, 3, 7], device="cuda", dtype=torch.int32),
        torch.tensor([0, 3], device="cuda", dtype=torch.int32),
        torch.tensor([0], device="cuda", dtype=torch.int64),
        torch.tensor([True], device="cuda"),
        history,
        torch.zeros(2, device="cuda", dtype=torch.int32),
        tables,
        spec=spec,
        solution="cutedsl",
        enable_pdl=False,
    )

    assert output[:, 0].float().cpu().tolist() == expected_rows
    assert history[0, :3].cpu().tolist() == [5, 3, 7]
