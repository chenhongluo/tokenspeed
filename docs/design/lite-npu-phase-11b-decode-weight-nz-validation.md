# Lite NPU Phase 11B: Decode-only Weight-NZ validation

## 1. Validation scope

Phase 11B prepares only the explicitly selected Lite Decode GEMM weights in Ascend format 29 after
checkpoint loading. Prefill weights remain in ND format. Routed-expert GMM, graph and overlap policy,
model mathematics, cache ownership, checkpoint mapping, OE host residency, and Phase 11A fused
Add-RMSNorm are unchanged.

The submitted implementation chain is:

| Purpose | Commit |
| --- | --- |
| Production design | `b64b2018` |
| Decode-only implementation | `b4d1811f` |
| Target runtime-config restore | `eb7e768e` |

The final fix records an exact target-wheel ABI finding: `torch.npu.config.allow_internal_format` is
write-only in that runtime. The implementation therefore restores the known previous/default value by
assignment in `finally`; it does not inspect or delete the property. The converted tensor remains format
29 after restoration.

## 2. Exact source and focused gates

The final direct-board and service validation used:

```text
commit:  eb7e768ef057ebdeeed079bc6a31f7ddad7164bb
archive: 23f95b347dd402cce29f4c0c539c3e4b7ed799598e96a3d06c399b0635545f9a
```

Local, tracking, remote branch, source archive, and deployed source were checked independently.
Validation used one node with 16 Atlas 910B devices, CANN 9.0, and Torch-NPU 2.9.0.post2. Existing
node-local dependencies were reused.

| Gate | Result |
| --- | --- |
| Local Weight-NZ focused tests | `17 passed`, `1 skipped` |
| Cumulative Lite tests | `123 passed`, `40 skipped`, `2 known unrelated failures` |
| NPU kernel shape/config/graph tests | `13 passed` |
| NPU runtime and launcher tests | `18 passed` |
| Full pre-commit | passed |

The two cumulative failures pre-existed this phase: one fake model lacks the checkpoint probe contract,
and one CPU-only case trips the existing platform guard. Neither calls the Weight-NZ path.

## 3. Post-load and direct-board result

Production `_Weight` objects were exercised after the real post-load hook. Prefill standard and
transposed candidates both remained format 2 and unprepared. Decode standard and transposed candidates
both became format 29; the transposed candidate switched its forward ABI only after successful
conversion. Outputs were finite and matched their ND references.

Each Decode rank prepares exactly 105 tensors: 21 KDA output projections, 28 MLA projections, 28
Grouped-MoE output projections, and 28 shared-expert down projections. Each Prefill rank prepares zero.
The NPUGraph board covered all six distinct production GEMM shapes at BS1 and BS2. All 12 capture/replay
cases passed with changed replay inputs, and the largest absolute difference from the ND reference was
`0.00390625`.

The strongest leaf result remained MLA q-a and kv-a: transposed NZ storage reduced the isolated GEMM
time by approximately 18%--21%. Other selected shapes improved by approximately 0%--2%. These are leaf
results only.

## 4. Exact 8P8D service admission

Both service runs used the same exact source and topology. The only A/B difference was the Decode
Weight-NZ flag.

| Property | Prefill | Decode |
| --- | --- | --- |
| Devices | 8 | 8 |
| Weight-NZ | disabled | A/B off or on |
| Graph | disabled | BS1/BS2 |
| Overlap | disabled | enabled |
| Model length / total tokens | 4096 / 8192 | 4096 / 8192 |
| Max batch | 2 | 2 |

All 16 workers loaded, both role-local services reached `SERVING`, and Decode captured `[1, 2]` once.
Both A/B runs passed the following request gates:

- 4-token smoke returned HTTP 200;
- exact 1,100-token prompt completed Prefill continuation and returned 4 tokens;
- six standalone stable-B requests had one digest;
- late-admission B and post-BS2 B matched the stable digest;
- smoke, long continuation, late-A, late-B, and post-B digests matched across off/on;
- logs had zero Traceback, RuntimeError, NotImplementedError, HTTP 5xx, or non-finite matches.

## 5. Service performance A/B

The fixed 64-token request used six warmed runs per side. Reported values are medians.

| Metric | Weight-NZ off | Weight-NZ on | On vs off |
| --- | ---: | ---: | ---: |
| Latency | 12.0732 s | 12.1515 s | 0.65% slower |
| TTFT | 0.5438 s | 0.5778 s | 6.26% slower |
| TPOT | 0.18533 s | 0.18652 s | 0.64% slower |
| Output rate | 5.3958 tok/s | 5.3614 tok/s | 0.64% lower |

The short-request TTFT is noisy, and the Decode-dominated TPOT difference is below 1%. This A/B shows
no material slowdown, but it also shows no measurable service speedup. The direct-board q-a/kv-a gain
does not carry through the current full-service critical path.

## 6. GSM8K first100 exact-source A/B

Both runs used EvalScope 1.9.1, GSM8K `main/test` first 100, 4-shot, seed 42, API batch 2, greedy
sampling, 512 output tokens, the real `"}\n\n"` stop sequence, `ignore_eos=true`, and
`no_stop_trim=true`. Each side used a fresh result directory without prediction reuse.

| Metric | Weight-NZ off | Weight-NZ on |
| --- | ---: | ---: |
| Completed | 100/100 | 100/100 |
| API error / empty output | 0 / 0 | 0 / 0 |
| Accuracy | 16/100 | 14/100 |
| Mean input tokens | 652.53 | 652.53 |
| Mean output tokens | 204.41 | 195.83 |
| Mean latency | 41.2264 s | 39.8322 s |
| Mean output throughput | 4.96 tok/s | 4.92 tok/s |

Prompts matched in 100/100 cases. Full outputs matched in 53/100, extracted answers in 66/100, and
judgments in 96/100. Of the four judgment differences, three were correct only with Weight-NZ off and
one only with Weight-NZ on.

This model and topology already exhibit low-order BF16/collective variability between otherwise equal
service starts. The bidirectional four-sample change is consistent with that established variability:
it does not establish a two-point Weight-NZ regression or improvement. It does establish 100/100
completion with no new API or empty-output failure.

## 7. Artifacts and cleanup

The three service artifact trees have aggregate SHA256:

```text
500cc209277b3b4784c6bda4306ff9f09a40d58ff3c22dc9e703a4f937f9f538
```

Private machine and checkpoint paths remain only in local artifacts and are not part of this commit.
After collection, the local tunnel and exact remote service process groups received TERM. The final
process-group checks, service listeners, tunnel listener, and all 16 device-file holder checks were zero.

## 8. Decision and rollback

Phase 11B is admitted as a bounded, Decode-only, reversible compatibility optimization. It passes exact
post-load, eager/graph, 8P8D functional, and first-100 correctness gates without an attributable
regression or material slowdown. The validation does not claim an end-to-end performance gain.

Set `LITE_DECODE_WEIGHT_NZ=0` to retain canonical ND Decode weights without changing graph, overlap, or
Phase 11A. A failed requested conversion continues to stop model loading rather than silently selecting
a different layout.
