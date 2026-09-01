# Lite NPU Phase 12A: merged KDA input projection validation

## 1. Validation scope

Phase 12A replaces the six rank-local Lite KDA hidden-state projections with one BF16 projection and
one explicit contiguous QKV materialization. The checkpoint still exposes the original six tensors;
the strict loader writes them into one `[2304,3072]` parameter by fixed component offsets.

The submitted chain is:

| Purpose | Commit |
| --- | --- |
| Phase 12 overall design | `cc84198a` |
| Phase 12A production design | `342ee84b` |
| Merged projection implementation and tests | `5da19b9b` |

The implementation does not add a runtime flag, duplicate parameter, kernel, dependency, graph path,
cache, or communication. `forget_b` and `beta_b` remain separate because they consume the two
128-dimensional low-rank projections.

## 2. Exact source and local gates

The off/on comparison used adjacent exact source commits:

| Side | Commit | Git archive SHA256 |
| --- | --- | --- |
| Off | `342ee84b59cf105c1c5b3c770c1f5b7beeb3a8ae` | `46287241464bc700341fd794b189fc7c53420eb2da54c4369ce6d9434a354e45` |
| On | `5da19b9b44f229aa9cb6a40c91b20cf93de5e9a1` | `7199e42a7c6ce8fd0224090dfcc67e21dbee7ca593a3f93a2d60d9c7f7340a35` |

Local, tracking, remote branch, archive, and deployed source were checked independently. Existing
node-local dependencies and the previously validated public KDA artifact were reused without package
installation.

| Gate | Result |
| --- | --- |
| Merged projection focused tests | `28 passed`, `18 skipped` |
| Cumulative Lite tests | `133 passed`, `43 skipped`, `2 known unrelated failures` |
| Exact-source NPU focused tests | `46 passed`, `2 warnings` |
| Full pre-commit before implementation commit | passed |

The two cumulative failures pre-existed Phase 12A: one fake model lacks the checkpoint probe contract,
and one CPU-only case reaches the existing accelerator platform guard. Neither calls the merged
projection path.

The loader tests prove all six component writes, TP1/TP8 geometry, replicated low-rank pieces, strict
missing/duplicate/wrong-shape/wrong-dtype rejection, meta construction, exact target coverage, absence
of the old six registered parameters, and unchanged total parameter elements.

## 3. NPU projection and graph board

The exact production shape used BF16 hidden input `[T,3072]` and the rank-local merged weight
`[2304,3072]`. The baseline includes six GEMMs and QKV concatenation; the candidate includes one GEMM
and the production QKV contiguous operation.

| Tokens | Six projections (us) | Merged projection (us) | Speedup | Relative L2 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 53.71 | 27.83 | 1.930x | exact |
| 2 | 54.90 | 36.27 | 1.514x | exact |
| 32 | 53.95 | 34.72 | 1.554x | `7.996e-9` |
| 256 | 63.52 | 34.11 | 1.862x | `2.926e-5` |
| 1024 | 93.37 | 63.83 | 1.463x | `2.832e-5` |

All outputs were finite and QKV was contiguous. Decode BS1 and BS2 NPUGraph tests captured and
replayed changed inputs with stable output addresses and finite results. T1/T2 were elementwise exact;
larger shapes remained below the frozen `1e-4` relative-L2 limit.

## 4. Exact 8P8D service admission

Both runs used the same model, request corpus, cache limits, ports, and cumulative topology. The only
source difference was the Phase 12A implementation.

| Property | Prefill | Decode |
| --- | --- | --- |
| Devices | 8 | 8 |
| Graph | disabled | BS1/BS2 |
| Overlap | disabled | enabled |
| Weight-NZ | disabled | enabled |
| Model length / total tokens | 4096 / 8192 | 4096 / 8192 |
| Max batch | 2 | 2 |

Both sides reached ready and passed:

- HTTP 200 and non-empty four-token smoke;
- six fixed 64-token requests;
- an exact 1,100-token Prefill-continuation request;
- repeated stable-B requests and post-BS2 state-isolation replay;
- real late-admission BS2;
- retraction followed by an exact stable-B recovery;
- identical fixed request and response digests across off/on.

Decode captured BS1/BS2 once. Both sides had zero graph fallback and zero fatal or non-finite log
matches.

## 5. Fixed-request performance

The fixed 64-token request used six warmed runs per side. Values are medians.

| Metric | Off | On | On vs off |
| --- | ---: | ---: | ---: |
| Latency | 12.057715 s | 12.194339 s | 1.13% slower |
| TTFT | 0.582660 s | 0.555342 s | 4.69% faster |
| TPOT | 0.184992 s | 0.187729 s | 1.48% slower |
| Output rate | 5.224875 tok/s | 5.166339 tok/s | 1.12% lower |

The complete-service difference is small and mixed: TTFT improves while TPOT moves by 1.48%. This is
within the observed run-to-run noise of the distributed service, so Phase 12A claims no end-to-end
speedup. The leaf projection speedup remains real but is a small part of the full token critical path.

## 6. GSM8K first100 A/B

Both fresh runs used EvalScope 1.9.1, GSM8K `main/test` first 100, 4-shot, seed 42, API batch 2, greedy
sampling, 512 output tokens, the real `"}\n\n"` stop sequence, `ignore_eos=true`, and
`no_stop_trim=true`.

| Metric | Off | On |
| --- | ---: | ---: |
| Completed | 100/100 | 100/100 |
| API error / empty output | 0 / 0 | 0 / 0 |
| Accuracy | 14/100 | 16/100 |
| Mean input tokens | 652.53 | 652.53 |
| Mean output tokens | 192.87 | 202.52 |
| Mean latency | 36.8784 s | 38.7745 s |
| Mean output throughput | 5.23 tok/s | 5.22 tok/s |

Full outputs matched in 48/100 cases, extracted answers in 63/100, and judgments in 98/100. The only
judgment differences were indices 49 and 58, both correct only with the merged projection; there was no
off-only correct sample.

The service already exhibits low-order BF16 and collective variability across otherwise equal starts.
The two unidirectional judgment changes do not establish a model-quality gain, but they exclude an
attributable first-100 accuracy regression. Both sides completed the full frozen corpus without an API
or empty-output failure.

The on run generated 5% more output tokens, so its larger mean latency and wall time cannot be treated
as an equal-work performance regression. Its output throughput remained within 0.2% of off.

## 7. Memory, logs, and cleanup

The merged parameter has exactly the same 7,077,888 BF16 elements per KDA layer as the six removed
parameters. Runtime HBM confirms no duplicate copy:

| Role | Off control / ordinary ranks | On control / ordinary ranks |
| --- | ---: | ---: |
| Prefill | 19,754 / 19,308--19,313 MiB | 19,738 / 19,292--19,298 MiB |
| Decode | 19,540 / 19,116--19,123 MiB | 19,558 / 19,134--19,141 MiB |

The off and on evidence-manifest SHA256 values are:

```text
off 000a0443f27d02047ef7a773e2c568cc6b45d427ee7bc92f8cc594e330716ba5
on  14f5eb01d57c2dad3beed2d7e40053f5a90b7b19d12e79a5ef085493533305c3
```

Private machine, checkpoint, prompt, and raw-log paths remain outside the repository. After evidence
collection, the exact local tunnel and remote service process groups received TERM. Final checks found
zero process-group, source-marker, run-marker, or NPU process rows; all nine role-local service ports
and the tunnel port had no listener.

## 8. Decision and rollback

Phase 12A is admitted. It preserves checkpoint coverage and parameter count, passes CPU/meta/NPU and
graph checks, speeds up the isolated production projection shape, completes the exact 8P8D admission
matrix, and has no attributable correctness or material service-performance regression.

There is intentionally no runtime branch. Rollback is the single implementation commit
`5da19b9b44f229aa9cb6a40c91b20cf93de5e9a1`; reverting it restores the six original parameters and
projection calls. Phase 12B may change only the remaining QKV materialization boundary and must retain
this loader layout unless its own evidence rejects the merged path.
