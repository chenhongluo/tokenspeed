# Lite NPU Phase 12B: packed causal-convolution negative validation

## 1. Result

Phase 12B is rejected as a production optimization. Removing the visible QKV materialization does not
cross the frozen production Decode graph gate, and the Prefill copy is too small to justify extending
the pinned public operator ABI.

No executable source, runtime flag, alternate kernel, checkpoint rule, state layout, or test-only
production branch is added. The admitted Phase 12A path remains unchanged.

## 2. Exact source and operator audit

The board used the exact Phase 12A executable source:

```text
commit:  5da19b9b44f229aa9cb6a40c91b20cf93de5e9a1
archive: 7199e42a7c6ce8fd0224090dfcc67e21dbee7ca593a3f93a2d60d9c7f7340a35
```

The Phase 12B design is `7070a1fe`. It changes documentation only, so its executable source is
identical. Tests ran on Atlas 910B with CANN 9.0 and Torch-NPU 2.9.0.post2. The public KDA artifact
remained pinned to vLLM-Ascend source `d543ccee0a1ff677165777e3defafd42b35e83ef`.

The source audit establishes why passing a `[T,1536]` view with physical row stride 2304 is not a new
Prefill kernel:

- the public op declares `AutoContiguous()` for input `x`;
- the binding requires input and weight channel dimensions to match;
- the kernel derives `dim=1536` and reads token rows at `token*dim+channel`;
- output is allocated with the input's logical shape.

A direct stride-aware implementation would therefore need a new input-stride contract, separate
logical channel and physical-stride tiling values, changed input offsets, changed output-shape logic,
binding/schema tests, and a rebuilt pinned artifact. It is not a one-line kernel reuse.

## 3. Prefill board

The baseline explicitly materializes QKV before the public convolution. The diagnostic passes the
strided view and lets the existing `AutoContiguous` contract handle it.

| Tokens | Temporary | Copy only | Explicit + conv | Strided + conv | Strided delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 96 KiB | 7.298 us | 1437.449 us | 1439.751 us | 0.16% slower |
| 256 | 768 KiB | 7.878 us | 1433.456 us | 1432.210 us | 0.09% faster |
| 1024 | 3 MiB | 7.277 us | 1445.718 us | 1457.807 us | 0.84% slower |

Output and final convolution state were elementwise exact for all three shapes. The copy accounts for
only 0.50%--0.55% of explicit copy-plus-convolution time, far below the 3% gate in every case. The
strided diagnostic stays within the 2% slowdown limit but provides no consistent speedup.

## 4. Operator profile

The T256 explicit trace contains a QKV-specific sequence:

```text
aten::contiguous -> aten::clone -> aten::copy_ -> aclnnInplaceCopy
```

Its device copy is 4.92 microseconds for `[256,1536]`. The strided trace has no separate QKV Aten
copy sequence. Both traces retain the unrelated common state-compaction and weight-transpose copies.

The public operator's device self time was 33.762 microseconds with explicit QKV and 36.582
microseconds with the strided view. Because the kernel source itself has no row-stride argument, the
platform's stride handling is subsumed in the public-op boundary rather than exposed as a separate
Aten event. Moving the visible copy into that boundary does not remove the required layout
conversion.

The two `operator_details.csv` files have SHA256:

```text
explicit 41df329c80d7c0fac856851cb34fc7e43eb57ceb026193dcea351f74987d7c72
strided  af6fcd68b803e94e0f73c54f35ab6328216d3a9b7fe6f49a4857d7cf9133e189
```

Raw profiler output and private machine metadata remain outside the repository.

## 5. Decode eager and graph board

| Mode | Shape | QKV contiguous before call | Temporary | Explicit | Strided | Strided delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Eager | BS1 | yes | 0 | 198.281 us | 198.419 us | 0.07% slower |
| Eager | BS2 | no | 6 KiB | 208.406 us | 200.288 us | 3.90% faster |
| Graph | BS2 | no | 6 KiB | 82.995 us | 82.624 us | 0.45% faster |

Outputs and final states were elementwise exact. BS2 graph captured both candidates, replayed changed
input and page indices, retained stable output addresses, and produced finite values.

The production graph result was repeated in 11 alternating paired trials with 500 replays per side:

| Metric | Explicit | Strided |
| --- | ---: | ---: |
| Median | 81.780 us | 81.435 us |
| Range | 81.722--81.958 us | 81.421--81.530 us |
| Candidate delta |  | 0.42% / 0.345 us faster |

The ranges do not overlap, so the sub-microsecond difference is measurable at the leaf. It still
fails both production thresholds: 3% and 2 microseconds per KDA layer. Across all 21 KDA layers the
maximum measured saving is about 7.2 microseconds per Decode step, which is negligible relative to
the approximately 188-millisecond service TPOT measured in Phase 12A.

The larger eager BS2 percentage is not used for admission because the supported Decode service uses
BS1/BS2 graph. BS1 cannot benefit: a one-row slice is already contiguous and performs no copy.

## 6. Decision against each gate

| Gate | Result |
| --- | --- |
| Output/state exact and finite | passed |
| Changed-input BS2 graph replay | passed |
| Stable output address | passed |
| Prefill diagnostic slowdown below 2% | passed |
| Prefill copy at least 3% in two shapes | failed; below 0.56% in all shapes |
| Decode graph at least 3% faster | failed; 0.42% |
| Decode graph at least 2 us/layer faster | failed; 0.345 us |

The decision stops before production integration. Consequently there is no new CPU/NPU regression
surface and no reason to repeat the 16-rank service or GSM8K: the deployed executable remains the
already admitted Phase 12A commit.

## 7. Cleanup and future threshold

Both profiling processes exited normally. Final hardware inspection found no running process on any
of the 16 devices. No service, listener, cache, checkpoint, or source deployment was created for this
negative experiment.

Reconsider a stride-aware public kernel only if a later merged projection changes the copy to at
least 3% of the Prefill boundary, or if a larger supported Decode graph batch makes the saved copy at
least 2 microseconds and 3% per layer. Until then, the explicit Phase 12A materialization is the
smaller and more observable implementation.
