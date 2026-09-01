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

import json
import re
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "design"
    / "lite-npu-public-kernels.json"
)
REQUIRED_CAPABILITIES = {
    "attention_partial_state_merge",
    "embedding_dispatch",
    "fused_add_rmsnorm",
    "grouped_moe_combine",
    "grouped_moe_dispatch",
    "grouped_moe_expert_gmm_swiglu",
    "grouped_moe_topk",
    "kda_decode_recurrent",
    "kda_gate_cumsum",
    "kda_output_epilogue",
    "kda_prefill_chunk_scan",
    "lite_featurewise_beta_prepare",
    "nope_mla_attention_lse",
    "nope_mla_prolog",
    "oe_block_projection_host_lookup",
    "packed_qkv_causal_conv",
    "paged_mha",
    "rotary_embedding",
}


def test_lite_npu_public_kernel_manifest():
    text = MANIFEST.read_text()
    manifest = json.loads(text)

    assert manifest["schema_version"] == 1
    assert not re.search(r"(?:10|127|192\.168)\.\d+\.\d+\.\d+", text)
    assert not re.search(r"/home/|ssh://|password|passwd|job[_-]?id", text, re.I)

    kernels = manifest["kernels"]
    capabilities = [kernel["capability"] for kernel in kernels]
    assert len(capabilities) == len(set(capabilities))
    assert REQUIRED_CAPABILITIES <= set(capabilities)

    for kernel in kernels:
        assert kernel["status"] in {"existing", "candidate", "adapt", "missing"}
        assert kernel["integration"] in {"reuse", "thin-adapter", "incremental", "new"}
        assert set(kernel["roles"]) <= {"prefill", "decode"}
        assert kernel["roles"]
        for field in ("covered_steps", "shape_constraints", "validation"):
            assert kernel[field] and all(
                isinstance(value, str) for value in kernel[field]
            )

        provenance = (
            kernel["source_url"],
            kernel["revision"],
            kernel["license"],
            kernel["upstream_path"],
        )
        if kernel["status"] == "missing":
            assert provenance == (None, None, None, None)
            continue

        assert all(provenance)
        assert urlparse(kernel["source_url"]).scheme == "https"
        assert urlparse(kernel["source_url"]).hostname == "github.com"
        assert re.fullmatch(r"[0-9a-f]{40}", kernel["revision"])
        path = PurePosixPath(kernel["upstream_path"])
        assert not path.is_absolute() and ".." not in path.parts

    baseline = manifest["baseline"]
    assert urlparse(baseline["source_url"]).hostname == "github.com"
    assert re.fullmatch(r"[0-9a-f]{40}", baseline["revision"])
    assert baseline["license"]
