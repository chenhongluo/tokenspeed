# Lite NPU Phase 11A: fused Add-RMSNorm validation

## 1. Validation scope

Phase 11A changes only Lite's post-attention residual boundary from a BF16 add followed by standalone
RMSNorm to the existing unified residual RMSNorm API. Prefill remains eager and overlap-off. Decode
remains graph BS1/BS2 with overlap enabled. The MLP tail residual, model topology, checkpoint mapping,
PD transfer, cache layout, routing, sampling, and service limits are unchanged.

The submitted chain is:

| Purpose | Commit |
| --- | --- |
| Production design | `a68cf2b0` |
| Lite decoder call site | `af733171` |
| Lite `_Norm` residual ABI | `3f934f16` |
| Alias-preserving NPU oracle | `1676360b` |

The latter two commits are exact-source findings rather than scope expansion: the first remote model
test exposed that Lite's local norm helper did not yet forward the unified residual argument, and the
next test exposed an oracle that cloned the intentionally mutated residual input too late. No kernel,
dependency, feature flag, or cross-layer residual carrier was added.

## 2. Exact source and focused gates

The final service used this tracked source:

```text
commit:  1676360b14cb29d6daa417827149aa5bcb07c674
archive: 63e88a556d9aafa34356629289abfa4f8aa98d9d7784bdfad15e9e7bc7b8b57a
```

Local, tracking, remote branch, archive, and deployed source were checked independently. Validation
used one node with 16 Atlas 910B devices and CANN 9.0. Existing node-local runtime dependencies were
reused; the system environment was not modified.

| Gate | Result |
| --- | --- |
| Lite decoder focused | `6 passed`, `1 skipped`, `4 deselected` |
| Frozen reference replay | `36 passed`, `2 skipped` |
| NPU kernel eager and graph | `6 passed` |
| NPU Lite model residual cases | `7 passed` |
| Full pre-commit | passed |

The NPU kernel cases covered BF16 and FP32 eager execution plus graph BS1/BS2. Lite model cases checked
finite output, the fused numerical oracle, and exact BF16 residual publication through the existing
alias-preserving wrapper.

## 3. Direct-board result

The direct board tested Lite's hidden width at token counts 1, 2, 32, and 1024. All BF16 outputs were
finite, residual sums were elementwise equal to standalone BF16 addition, and graph capture/replay
passed. The target operator correctly rejected BF16 inputs paired with an FP32 norm weight.

The complete alias-preserving eager path was approximately flat for token counts 1--32 and improved
from `26.570 us` to `15.593 us` at 1024 tokens. NPUGraph replay was also approximately flat at BS1/BS2
and improved from `49.612 us` to `43.123 us` at 1024 tokens. These numbers establish a long-Prefill
kernel benefit; they do not establish an end-to-end Decode gain.

## 4. Exact 8P8D service admission

The service retained the Phase 10B role policy:

| Property | Prefill | Decode |
| --- | --- | --- |
| Devices | 8 | 8 |
| Graph | disabled | BS1/BS2 |
| Overlap | disabled | enabled |
| Prefix cache | disabled | n/a |
| Model length / total tokens | 4096 / 8192 | 4096 / 8192 |
| Max batch | 2 | 2 |

All 16 workers loaded the checkpoint, both role-local gRPC services reached `SERVING`, and the Gateway
returned exactly one `lite` model. Decode captured `[1, 2]` once during startup and did not recapture or
fall back during validation.

The request-level gates were:

- a 4-token smoke returned HTTP 200 with 4 completion tokens;
- an exact 1100-token prompt completed a `1024 + 76` Prefill continuation and returned 4 tokens;
- six standalone stable-B requests had one digest;
- late-admission B and post-BS2 B matched that digest;
- the stable-B digest also matched the Phase 10B service exactly;
- Prefill, Decode, Gateway, and launcher logs had zero Traceback, RuntimeError,
  NotImplementedError, NaN/Inf, or non-finite matches.

## 5. GSM8K first100 A/B

Both runs used EvalScope 1.9.1, GSM8K `main/test` first 100, 4-shot, seed 42, API batch 2, greedy
sampling, 512 output tokens, the real `"}\n\n"` stop sequence, `ignore_eos=true`, and
`no_stop_trim=true`. Phase 11A used a new result directory without prediction reuse.

| Metric | Phase 10B separate | Phase 11A fused |
| --- | ---: | ---: |
| Completed | 100/100 | 100/100 |
| API error / empty output | 0 / 0 | 0 / 0 |
| Accuracy | 13/100 | 15/100 |
| Mean input tokens | 652.53 | 652.53 |
| Mean output tokens | 192.85 | 198.94 |
| Mean latency | 39.2276 s | 40.1418 s |
| Mean output throughput | 4.92 tok/s | 4.96 tok/s |
| Wall time | 2025.69 s | 2077.38 s |

The two runs had 54/100 full outputs, 68/100 extracted answers, and 98/100 judgments equal. The two
judgment differences were indices 42 and 59, both correct only in the fused run. No sample was correct
only in the separate baseline.

This does not establish a two-point accuracy gain: earlier phases demonstrated low-order Prefill
collective variability on this checkpoint. It does establish that the fused boundary introduced no
attributable first-100 regression. The raw latency increased by 2.33% while generated tokens increased
by 3.16%; output throughput increased by 0.81%, and wall time per output token decreased by 0.59%.
These differences are small, so service A/B establishes no material slowdown but does not claim a
stable end-to-end speedup.

## 6. Artifacts and cleanup

The EvalScope result tree aggregate is:

```text
c2ed27de4f4fbe43388c29c521289a603348ad05991ca057de992bcdaa981c4a
```

The 11-file service artifact was copied back and independently hashed on both hosts:

```text
b8b4d6ca2ea11028f3d4075a61335146504965b703ac42eee36a65299f61169f
```

After collection, the local tunnel and exact remote service process group received TERM. The final
process group, source/run markers, fixed-port bind check, local tunnel listener, and all 16 device-file
holder checks were zero or free.

## 7. Decision

Phase 11A is admitted. It reuses the existing official fused operator at the minimum Lite boundary,
passes exact eager/graph numerical checks and the cumulative 8P8D service gates, and shows no
attributable correctness or material performance regression. The implementation remains enabled for
both Prefill and Decode. Phase 11B may now evaluate Decode-only Weight-NZ independently.
