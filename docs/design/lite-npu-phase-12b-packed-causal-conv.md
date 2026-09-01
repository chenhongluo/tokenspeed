# Lite NPU Phase 12B: packed causal-convolution experiment

## 1. Scope

Phase 12B decides whether the QKV view produced by the Phase 12A merged projection can reach the
existing packed causal-convolution without a separate materialization.

It does not change KDA mathematics, projection weights, checkpoint loading, packed convolution
weights, convolution state layout, recurrence, graph ownership, cache, communication, or the 8P8D
topology. A candidate that does not cross the frozen performance gate ends as a negative validation;
no production branch or dead kernel is retained.

## 2. Current boundary

For TP8, the merged projection produces contiguous BF16 `[T,2304]`. QKV occupies its first 1536
columns:

```text
projected = hidden @ W_merged^T  # shape [T,2304], stride [2304,1]
qkv_view  = projected[:, :1536] # shape [T,1536], stride [2304,1]
```

`qkv_view` is contiguous only when `T=1`; it is not row-contiguous for `T>1`. The current explicit
copy allocates:

```text
temporary bytes = T * 1536 * sizeof(BF16) = T * 3072
```

This is 6 KiB for Decode BS2, 96 KiB for T32, 768 KiB for T256, and 3 MiB for T1024.

The rest of the path is already packed:

- checkpoint post-load owns one `[1536,4]` QKV convolution weight;
- cache owns one `[pages,1536,3]` packed convolution state;
- each layer issues one convolution call;
- output is one contiguous `[T,1536]` tensor.

There is no three-call or three-cache conversion left to optimize.

## 3. Existing implementations

### 3.1 Prefill

The public Ascend `CausalConv1d` declares `AutoContiguous()` for input `x`. Its kernel indexes the
input as:

```text
x_offset = token * dim + channel
dim      = 1536
```

It therefore cannot directly interpret the merged view's 2304-element row stride. Passing the view
to the current adapter is a diagnostic only: the platform must materialize an equivalent contiguous
input before the kernel reads it.

A truly stride-aware Prefill candidate would need an ABI and tiling change that separately carries
the logical channel count 1536 and physical row stride 2304. The kernel would use the physical stride
only for input loads while retaining the current contiguous output and state layouts. This is not a
generic strided-tensor framework.

### 3.2 Decode

The current graph-safe Decode path constructs one convolution window with `torch.cat(history,
projected.unsqueeze(-1))`. `torch.cat` can read the strided QKV view directly while producing its
already-required contiguous window. Decode therefore has a zero-ABI candidate: omit the preceding
QKV copy and let the existing window construction consume the view.

BS1 is already contiguous by PyTorch's size-one stride rule, so only BS2 can benefit under the
supported graph configuration.

## 4. Experiment matrix

Use the exact Phase 12A implementation and target BF16 shapes on one Atlas 910B device.

| Role | Shapes | Baseline | Candidate |
| --- | --- | --- | --- |
| Prefill | T32/T256/T1024 | explicit QKV copy + public conv | strided view + current `AutoContiguous` conv |
| Decode eager | BS1/BS2 | explicit QKV copy + tensor conv | strided view + tensor conv |
| Decode graph | BS2 | captured explicit copy + tensor conv | captured strided view + tensor conv |

For every shape, record:

- whether the source view is contiguous;
- exact temporary bytes;
- standalone copy time;
- copy-plus-convolution or graph-replay time;
- output and final-state maximum absolute difference;
- changed-input graph replay and stable output address;
- operator profile entries for explicit and adapter-owned copies.

Decode graph timing uses at least nine alternating paired trials with at least 500 replays per side.
Alternating order avoids treating device-temperature or launch-order drift as a candidate benefit.

## 5. Admission gates

All correctness gates are hard:

- output and final convolution state are elementwise exact;
- all values are finite;
- BS2 graph replays changed inputs and pages without fallback;
- output addresses remain stable;
- packed weight, state, and output layouts remain unchanged.

The performance gate reflects the maximum possible end-to-end value of removing this one copy:

- Decode BS2 graph median must improve by both at least 3% and 2 microseconds per KDA layer;
- paired-trial ranges must not overlap;
- Prefill copy must account for at least 3% of copy-plus-public-convolution time in at least two of
  T32/T256/T1024 before a new public stride ABI is considered;
- the strided diagnostic must not make any Prefill shape more than 2% slower.

The two-part Decode threshold avoids adding a role branch for a statistically visible but
end-to-end-negligible sub-microsecond result. Across 21 KDA layers, a 2-microsecond leaf gain is at
most 42 microseconds per token before other work; smaller results cannot materially move the current
service TPOT.

## 6. Production decision tree

Stop at the first applicable result:

1. If Decode does not cross its graph gate, retain the explicit Phase 12A copy everywhere.
2. If Decode crosses the gate but Prefill does not, return a view from the merged projection and make
   only Prefill materialize it before the backend call.
3. If Prefill also crosses its gate, prototype the minimal row-stride extension in the pinned public
   operator and its existing binding; do not add a second operator or state layout.
4. Admit code only after CPU shape/contiguity tests, NPU eager/graph exactness, cumulative tests, and
   exact 8P8D service A/B. Otherwise commit only the negative validation.

No runtime flag is needed. A successful code change is reverted as one implementation commit. A
negative result keeps the Phase 12A implementation commit as the production rollback point and does
not require another 8P8D or GSM8K run because no executable source changes.
