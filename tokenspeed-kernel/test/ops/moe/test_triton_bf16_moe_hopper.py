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

"""Exercise pointer-gather BF16 MoE on SM90, including masked routing tails."""

from types import SimpleNamespace

import pytest
import torch

if (
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability()[0] != 9
):
    pytest.skip("requires NVIDIA Hopper (SM90)", allow_module_level=True)

from tokenspeed_kernel.ops.moe.triton.bf16 import (  # noqa: E402
    triton_bf16_precomputed_moe_apply,
)


@pytest.mark.parametrize("num_tokens", [1, 17, 65])
@pytest.mark.parametrize("intermediate_size", [32, 64])
def test_bf16_moe_hopper_gather_matches_reference(
    monkeypatch, num_tokens, intermediate_size
):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    generator = torch.Generator(device="cuda").manual_seed(42)

    def normal(*shape):
        return torch.randn(
            *shape, generator=generator, dtype=torch.bfloat16, device="cuda"
        )

    hidden_size, num_experts, topk = 128, 4, 3
    x = normal(num_tokens, hidden_size)
    w13 = normal(num_experts, 2 * intermediate_size, hidden_size) * 0.05
    w2 = normal(num_experts, hidden_size, intermediate_size) * 0.05
    ids = torch.arange(num_tokens * topk, device="cuda", dtype=torch.int32)
    ids = (ids % 3).reshape(num_tokens, topk)  # expert 3 has no routes
    ids[::2, -1] = -1  # identity-expert placeholders are zero in this kernel
    weights = torch.full((num_tokens, topk), 1 / topk, device="cuda")
    module = SimpleNamespace(w13_weight=w13, w2_weight=w2, ep_size=1)
    actual = triton_bf16_precomputed_moe_apply(
        {"activation": "silu"}, x, module, None, weights, ids
    )

    # Match the kernel's BF16 intermediate/route-output storage boundaries.
    expected = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(num_experts):
        tokens, slots = torch.where(ids == expert)
        if tokens.numel() == 0:
            continue
        projected = torch.nn.functional.linear(x[tokens].float(), w13[expert].float())
        gate, up = projected.chunk(2, dim=-1)
        intermediate = (torch.nn.functional.silu(gate) * up).bfloat16()
        routes = torch.nn.functional.linear(
            intermediate.float(), w2[expert].float()
        ).bfloat16()
        expected.index_add_(0, tokens, routes.float() * weights[tokens, slots, None])
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.float(), expected.bfloat16().float(), rtol=0.03, atol=0.002
    )
