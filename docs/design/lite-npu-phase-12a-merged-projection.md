# Lite NPU Phase 12A: merged KDA input projection

## 1. Scope

Phase 12A replaces six Lite KDA hidden-state GEMMs with one rank-local BF16 GEMM:

```text
[Q | K | V | output_gate | forget_a | beta_a] = hidden @ W_merged^T
```

It applies to both Prefill and Decode and preserves every downstream KDA operation. The change is
limited to Lite checkpoint placement and `LiteKDAParameters`; it does not add a kernel, dependency,
server argument, runtime registry, quantization mode, graph executor, cache, or communication.

`forget_b` and `beta_b` remain separate GEMMs because they consume different 128-dimensional
low-rank activations. Phase 12B, not this phase, decides whether the remaining packed-QKV
materialization can be removed from the causal-convolution boundary.

## 2. Existing implementation to reuse

The Kimi KDA implementation already establishes the useful pattern:

- one registered merged projection weight;
- fixed component offsets;
- row-sharded per-head sources and a replicated low-rank source;
- one `F.linear` followed by views;
- strict post-load coverage of all source pieces.

Lite reuses that pattern but not the exact Kimi class. Kimi's final component is one scalar-beta row
per local head. Lite has two replicated 128-row hidden-state projections, `forget_a` and `beta_a`,
followed by two separate row-sharded up projections. Reusing the Kimi shape would silently load the
wrong beta semantics.

The existing Lite strict loader remains the sole checkpoint trust boundary. Its existing
`LiteWeightSpec.component_id` field already identifies component writes for OE projection, so this
phase reuses that field under a distinct KDA category rather than adding another spec type.

## 3. Exact layout

For the production configuration:

```text
hidden_size       = 3072
linear_num_heads  = 32
linear_head_dim   = 128
KDA TP            = 8
local_projection  = 32 * 128 / 8 = 512
merged_rows       = 4 * 512 + 2 * 128 = 2304
```

The rank-local weight is BF16 `[2304,3072]`:

| Component ID | Source suffix | Destination rows | Local source rule |
| ---: | --- | ---: | --- |
| 0 | `q_proj.weight` | `[0,512)` | source rows `rank*512:(rank+1)*512` |
| 1 | `k_proj.weight` | `[512,1024)` | source rows `rank*512:(rank+1)*512` |
| 2 | `v_proj.weight` | `[1024,1536)` | source rows `rank*512:(rank+1)*512` |
| 3 | `g_proj.0.weight` | `[1536,2048)` | source rows `rank*512:(rank+1)*512` |
| 4 | `f_proj.0.weight` | `[2048,2176)` | full replicated source |
| 5 | `b_proj.0.weight` | `[2176,2304)` | full replicated source |

The merged row count is exactly 18 blocks of 128. No padding or alignment-only rows are needed.
Static weight elements and bytes are unchanged:

```text
old = (4 * 512 + 2 * 128) * 3072 = 7,077,888 BF16 elements
new = 2304 * 3072                 = 7,077,888 BF16 elements
bytes per KDA layer               = 13.5 MiB
```

The old six parameters must be removed. Creating the merged tensor in a post-load hook while keeping
the originals would add 13.5 MiB/layer, or 283.5 MiB/rank across 21 KDA layers.

The remaining low-rank up projections are registered as:

```text
forget_b [512,128]
beta_b   [512,128]
```

Their checkpoint source names remain unchanged; only their runtime target names may be made explicit
to avoid retaining an otherwise empty `Sequential` container.

## 4. Model-local implementation

Add one small Lite-only module that owns the merged parameter, component offsets, load operation, and
projection split. It has exactly one implementation and no factory or registry.

Its forward path is:

```text
projected   = F.linear(hidden, merged_weight)     # [T,2304]
mixed_qkv   = projected[:, 0:1536].contiguous()  # [T,1536]
output_gate = projected[:, 1536:2048]             # [T,512]
forget_a    = projected[:, 2048:2176]             # [T,128]
beta_a      = projected[:, 2176:2304]             # [T,128]
forget      = F.linear(forget_a, forget_b)
beta_logits = F.linear(beta_a, beta_b)
```

The explicit QKV materialization is intentional. The public Ascend causal-convolution input uses
`AutoContiguous()`, so passing the `[T,2304]` slice would move the same copy behind the adapter. The
explicit copy replaces the current three-way QKV `torch.cat` and gives Phase 12B a measurable
boundary.

All existing probe meanings are retained:

- `kda.qkv-projection` observes the contiguous `[T,1536]` tensor;
- `kda.output-gate` observes the `[T,512]` view;
- `kda.forget-down` observes the `[T,128]` view;
- `kda.beta-down` observes the `[T,128]` view;
- `kda.beta-logits` remains the result of `beta_b`.

Empty-token forward returns before the merged GEMM, as it does today.

## 5. Strict loader changes

The checkpoint layout continues to enumerate the original six source names, shapes, and dtypes.
For those names only, it emits:

- one common merged target name;
- category `kda-packed-projection`;
- component IDs 0 through 5;
- the existing `linear` row shard for Q/K/V/output-gate;
- replicated placement for forget-a/beta-a.

The model loader obtains the normal local shard through its existing `_local_weight` function, then
delegates the component write to the merged module. It tracks the six component IDs per target and
adds the merged target to `loaded_targets` only after all six are present.

All existing trust-boundary behavior remains fail-closed:

- source names are still checked before any target write;
- duplicate source names are rejected;
- source shape and dtype are validated against the unchanged checkpoint schema;
- component shape and target shape are checked before copy;
- a missing component leaves the merged target missing at the final target-coverage check;
- meta parameters validate layout but skip data copy;
- no general model loader behavior changes.

This keeps source coverage and target coverage distinct: six source tensors prove checkpoint
completeness, while one registered target proves there is no duplicate resident weight.

## 6. Direct-board evidence

Two boards were run at the exact rank-local BF16 shape on Atlas 910B. The first isolated the merged
GEMM. The second included the production explicit QKV materialization in the merged path.

The full projection boundary result is:

| Tokens | Six GEMMs + QKV cat (us) | Merged + QKV contiguous (us) | Speedup | Max abs | Relative L2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.899 | 14.552 | 2.604x | 0 | 0 |
| 2 | 37.893 | 19.526 | 1.941x | 0 | 0 |
| 32 | 38.168 | 19.557 | 1.952x | 0.125 | `8.27e-6` |
| 256 | 59.300 | 22.424 | 2.645x | 0.5 | `2.22e-5` |
| 1024 | 89.392 | 54.388 | 1.644x | 0.5 | `2.91e-5` |

The materialization reduces but does not remove the projection benefit. Decode T1/T2 remains
elementwise exact. Larger token counts use a different valid BF16 GEMM tiling, so they require
relative-L2 and downstream checks rather than bitwise equality.

## 7. Validation matrix

### 7.1 CPU/meta

- TP1 and TP8 merged shape, offsets, dtype, and registered-parameter count;
- exact static element equality with the six old tensors and absence of old registered parameters;
- component writes for all six sources, including TP8 ranks 0 and 7;
- replicated forget-a/beta-a equality across ranks;
- missing, duplicate, unexpected, wrong-shape, wrong-dtype, invalid-component, and duplicate-component
  failures;
- meta construction and full expected-source/target coverage;
- projection split and beta/forget up-projection against separate linear references;
- explicit contiguous QKV, unchanged probe labels, empty-token behavior, and existing conv packing;
- cumulative Lite model/loader/cache/graph/Grouped-MoE/OE tests.

### 7.2 NPU leaf

- T1/T2/T32/T256/T1024 at `[T,3072]` with target BF16 weights;
- T1/T2 elementwise output equality and finite values;
- T32/T256/T1024 relative L2 no worse than `1e-4`;
- the full merged-plus-contiguous boundary remains faster than the six-GEMM baseline at every shape;
- Decode BS1/BS2 graph capture/replay with changed inputs and stable weight/output addresses;
- full KDA layer output and state against the preceding exact implementation;
- no hidden `torch.cat` in the KDA hidden-state projection path.

### 7.3 Exact 8P8D

Use the supported cumulative topology: P eager/overlap-off, D graph BS1/2 and overlap-on. Compare the
implementation exact commit with this design commit as the baseline using the same checkpoint,
launcher, prompts, and cache limits.

Required evidence includes readiness; four-token and 64-token smoke; a 1,100-token Prefill
continuation; stable BS1; late-admission BS2; retraction; finite/fatal/graph-fallback scans; KDA state
isolation; HBM; TTFT; TPOT; throughput; and frozen EvalScope GSM8K first 100 with per-sample
comparison. T1/T2 exactness at the projection leaf does not justify a claim that complete service
outputs are bitwise deterministic across separate distributed launches.

## 8. A/B and rollback

No runtime flag is added. Supporting both paths in one loaded model would either duplicate the six
old parameters or add loader and forward branches solely for an experiment. The A/B uses adjacent
exact source commits:

- off: the Phase 12A design commit;
- on: the Phase 12A implementation commit.

Rollback is the single implementation commit. The merged path is rejected if it introduces a
checkpoint coverage gap, extra resident weights, graph failure/fallback, non-finite value, KDA state
mutation, attributable GSM8K regression, or material end-to-end slowdown. Leaf speedup without
complete service admission is insufficient.
