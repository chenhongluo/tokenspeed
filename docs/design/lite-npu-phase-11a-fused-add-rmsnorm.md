# Lite NPU Phase 11A: fused Add-RMSNorm production design

## 1. Scope

Lite currently computes the post-attention boundary as two operations:

```text
residual = layer_input + attention_output
moe_input = RMSNorm(residual)
```

Phase 11A replaces only that boundary with the existing unified residual RMSNorm API:

```text
moe_input, residual = RMSNorm(attention_output, residual=layer_input)
```

The change applies to both Prefill and Decode. It does not add a kernel, dependency, command-line
option, cross-layer residual protocol, communication fusion, quantization, or graph mode. The MLP
output residual addition remains unchanged. This is the same boundary used by the reference Lite
implementation and avoids a broader Qwen-style decoder refactor that Lite does not need.

## 2. Existing implementation to reuse

TokenSpeed already has the kernel stack:

- Lite `_Norm` already delegates device execution to the unified `rmsnorm` API; Phase 11A extends its
  local signature with the API's existing `residual` argument;
- `tokenspeed-kernel` dispatches the residual form through the registered platform implementation;
- `tokenspeed-kernel-npu` calls `torch_npu.npu_add_rms_norm` and preserves the existing residual
  alias contract;
- Qwen2/Qwen3/DFlash already consume the same residual interface in production;
- Lite's Phase 8B replay already uses this exact interface as its fused oracle.

Therefore production requires one Lite norm signature extension, one model call-site change, and
focused tests. Runtime must not import `torch_npu`, and no new wrapper is introduced.

## 3. Operator contract

The target wheel exposes:

```text
npu::npu_add_rms_norm(
    Tensor x1,
    Tensor x2,
    Tensor gamma,
    float epsilon=1e-6,
) -> (Tensor normalized, Tensor rstd, Tensor residual_sum)
```

The official operator documentation defines `residual_sum = x1 + x2`, then normalizes that sum. On
Atlas A2 it supports BF16 and one-to-eight-dimensional ND tensors. `x1`, `x2`, and `gamma` must have
the same dtype; `gamma` covers the normalized trailing dimensions. Lite production uses:

| Field | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `x1` | `[T, 768]` | BF16 | attention output |
| `x2` | `[T, 768]` | BF16 | layer input residual |
| `gamma` | `[768]` | BF16 | post-attention RMSNorm weight |
| `normalized` | `[T, 768]` | BF16 | Grouped-MoE input |
| `residual_sum` | `[T, 768]` | BF16 | residual used after the MLP |

The target wheel rejects a BF16 input with FP32 `gamma`. The checkpoint-loaded production model and
the existing Phase 8B replay use BF16 norm weights; validation must assert this rather than silently
casting a weight in every layer. The public operator reference is:

<https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_add_rms_norm.md>

## 4. Numerical semantics

The fused kernel can normalize its internal addition result before the BF16 `residual_sum` is written.
The current separate path can only normalize the already rounded BF16 sum. Consequently, equality to
the separate path is not the acceptance criterion.

The existing exact-checkpoint Phase 8B replay established the required reference behavior:

- fused residual matched the frozen attention residual elementwise on all Prefill and Decode rows;
- Prefill MoE replay improved from 24/28 passing layers to 28/28;
- maximum Prefill MoE relative L2 decreased from `0.0701477` to `0.0044870`;
- Decode improved from 42/84 to 45/84 rows, confirming this is a real difference source although not
  the only Decode rounding source.

Thus Phase 11A intentionally adopts fused arithmetic. It must preserve finite output and the exact
BF16 residual while allowing normalized values to differ from the separate path.

## 5. Direct-board admission

The installed `torch_npu 2.9.0.post2` operator was tested on one Atlas 910B NPU with Lite's hidden
width. All BF16 cases were finite, graph capture/replay succeeded, and the fused residual was
elementwise equal to the standalone BF16 addition.

Eager measurements include the current TokenSpeed residual alias-preserving copy:

| Tokens | Fused (us) | Separate (us) | Separate / fused |
| ---: | ---: | ---: | ---: |
| 1 | 18.978 | 17.527 | 0.924x |
| 2 | 17.740 | 17.147 | 0.967x |
| 32 | 17.661 | 16.799 | 0.951x |
| 1024 | 15.593 | 26.570 | 1.704x |

NPUGraph replay measurements captured the complete alias-preserving fused or separate sequence. A
common input reset was present in both graphs:

| Tokens | Fused (us) | Separate (us) | Separate / fused |
| ---: | ---: | ---: | ---: |
| 1 | 22.843 | 21.981 | 0.962x |
| 2 | 21.808 | 22.587 | 1.036x |
| 32 | 22.010 | 26.181 | 1.190x |
| 1024 | 43.123 | 49.612 | 1.150x |

The isolated Decode BS1/BS2 differences are small and must not be advertised as a service-level gain.
The clear direct benefit is long Prefill. Final retention still requires an exact-source 8P8D service
A/B because graph scheduling, collectives, and the full layer dominate Decode latency.

## 6. Production data flow and probes

`LiteDecoderLayer.forward` keeps its existing input/output API:

1. save `layer_input` as `residual`;
2. standalone input RMSNorm;
3. execute MLA or KDA and the existing KDA output reduction;
4. call `post_attention_layernorm(attention_output, residual)`;
5. keep the returned normalized tensor as the MoE input;
6. keep the returned BF16 sum as the residual used by the final MLP addition.

Probe meanings remain unchanged:

- `attention.output`: unfused attention result;
- `attention.residual`: fused operator's BF16 sum;
- `moe.input`: fused operator's normalized output;
- `layer.output`: returned residual plus MoE output.

Idle forwards remain unchanged and do not invoke the fused operator.

## 7. Validation and rollback

The implementation commit must provide:

- CPU/model tests proving the same double-residual formula and unchanged P/D communication order;
- a negative dtype check or loaded-model assertion proving the NPU norm weight is BF16;
- NPU eager tests for Prefill-like and Decode-like shapes, finite values, exact residual, and numerical
  agreement with the fused FP32 oracle under the existing tolerance;
- NPU graph BS1/BS2 capture and replay;
- all cumulative Lite focused tests and full pre-commit;
- exact-source 8P8D smoke, late-admission BS2/state checks, and same-workload service A/B;
- the same GSM8K first-100 configuration used by Phase 10 before cumulative final acceptance.

The baseline is the previous exact commit; no runtime feature flag is added solely for A/B. If the
fused path causes a new service failure, non-finite value, graph fallback, attributable accuracy
regression, or material end-to-end slowdown, rollback is the single Lite call-site commit.
