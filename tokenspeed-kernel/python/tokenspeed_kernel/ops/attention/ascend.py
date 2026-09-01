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

"""Ascend attention kernel registrations."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_npu:
    from tokenspeed_kernel.ops.attention.kda_reference import (
        torch_kda_paged_prefill as _torch_kda_paged_prefill,
    )
    from tokenspeed_kernel_npu.ops.kda import (
        public_kda_causal_conv1d as _public_kda_causal_conv1d,
    )
    from tokenspeed_kernel_npu.ops.kda import (
        public_kda_paged_prefill as _public_kda_paged_prefill,
    )
    from tokenspeed_kernel_npu.ops.kda import (
        torch_kda_causal_conv1d as _torch_kda_causal_conv1d,
    )
    from tokenspeed_kernel_npu.ops.kda import (
        torch_kda_paged_decode as _torch_kda_paged_decode,
    )
    from tokenspeed_kernel_npu.ops.mha import (
        mha_decode_with_kvcache as _mha_decode_with_kvcache,
    )
    from tokenspeed_kernel_npu.ops.mha import (
        mha_extend_with_kvcache as _mha_extend_with_kvcache,
    )
    from tokenspeed_kernel_npu.ops.mha import mha_prefill as _mha_prefill

    _CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))
    _DTYPES = {torch.float16, torch.bfloat16}
    _OPTIONS = {
        "sliding_window": frozenset({False}),
        "support_sinks": frozenset({False}),
        "support_logit_cap": frozenset({False}),
        "return_lse": frozenset({False}),
    }

    @register_kernel(
        "attention",
        "mha_prefill",
        name="ascend_mha_prefill",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
        priority=Priority.PERFORMANT,
        traits=_OPTIONS,
        tags={"portability"},
    )
    def mha_prefill(**kwargs):
        return _mha_prefill(**kwargs)

    @register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="ascend_mha_extend_with_kvcache",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k_cache", "v_cache"), "dense", _DTYPES),
        priority=Priority.PERFORMANT,
        traits={
            **_OPTIONS,
            "page_size": frozenset({64, 128}),
            "is_causal": frozenset({False, True}),
        },
        tags={"portability"},
    )
    def mha_extend_with_kvcache(**kwargs):
        return _mha_extend_with_kvcache(**kwargs)

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="ascend_mha_decode_with_kvcache",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k_cache", "v_cache"), "dense", _DTYPES),
        priority=Priority.PERFORMANT,
        traits={
            **_OPTIONS,
            "page_size": frozenset({64, 128}),
            "q_len": frozenset({1}),
        },
        tags={"portability"},
    )
    def mha_decode_with_kvcache(**kwargs):
        return _mha_decode_with_kvcache(**kwargs)

    @register_kernel(
        "attention",
        "kda_paged_prefill",
        name="torch_ascend_kda_paged_prefill",
        solution="torch",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
        priority=Priority.REFERENCE,
        traits={
            "beta_mode": frozenset({"scalar", "featurewise"}),
            "recurrent_layout": frozenset({"k_major"}),
        },
        tags={"ascend", "reference"},
    )
    def kda_paged_prefill(**kwargs):
        return _torch_kda_paged_prefill(**kwargs)

    @register_kernel(
        "attention",
        "kda_paged_prefill",
        name="public_ascend_kda_paged_prefill",
        solution="public_kda",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
        priority=Priority.SPECIALIZED,
        traits={
            "beta_mode": frozenset({"featurewise"}),
            "recurrent_layout": frozenset({"k_major"}),
        },
        tags={"ascend", "throughput"},
    )
    def public_kda_paged_prefill(**kwargs):
        return _public_kda_paged_prefill(**kwargs)

    @register_kernel(
        "attention",
        "kda_paged_decode",
        name="torch_ascend_kda_paged_decode",
        solution="torch",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
        priority=Priority.PORTABLE,
        traits={
            "indexed_state": frozenset({True}),
            "single_token": frozenset({True}),
            "beta_mode": frozenset({"scalar", "featurewise"}),
            "recurrent_layout": frozenset({"k_major"}),
        },
        tags={"ascend", "portability"},
    )
    def kda_paged_decode(**kwargs):
        return _torch_kda_paged_decode(**kwargs)

    @register_kernel(
        "attention",
        "kda_causal_conv1d",
        name="torch_ascend_kda_causal_conv1d",
        solution="torch",
        capability=_CAPABILITY,
        signatures=format_signatures(("projected", "weight"), "dense", _DTYPES),
        priority=Priority.PORTABLE,
        traits={
            "forward_mode": frozenset({"prefill", "decode"}),
            "batch_class": frozenset({"small", "large"}),
            "activation": frozenset({"none", "silu"}),
            "width": frozenset({4}),
        },
        tags={"ascend", "portability"},
    )
    def kda_causal_conv1d(**kwargs):
        return _torch_kda_causal_conv1d(**kwargs)

    @register_kernel(
        "attention",
        "kda_causal_conv1d",
        name="public_ascend_kda_causal_conv1d",
        solution="public_kda",
        capability=_CAPABILITY,
        signatures=format_signatures(("projected", "weight"), "dense", _DTYPES),
        priority=Priority.SPECIALIZED,
        traits={
            "forward_mode": frozenset({"prefill"}),
            "batch_class": frozenset({"large"}),
            "activation": frozenset({"none", "silu"}),
            "width": frozenset({4}),
        },
        tags={"ascend", "throughput"},
    )
    def public_kda_causal_conv1d(**kwargs):
        return _public_kda_causal_conv1d(**kwargs)


__all__ = [
    "kda_causal_conv1d",
    "kda_paged_decode",
    "kda_paged_prefill",
    "mha_decode_with_kvcache",
    "mha_extend_with_kvcache",
    "mha_prefill",
]
