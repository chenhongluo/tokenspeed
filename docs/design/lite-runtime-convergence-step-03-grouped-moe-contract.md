# Flash-Lite GPU/NPU runtime convergence: Step 3 Grouped MoE contract

## Scope

This step moves the Ascend Grouped MoE implementation out of the temporary
`LiteForCausalLM` model and makes the shared FLASHLocal decoder select it at
construction time. The model-level contract is common:

```text
project input -> RMSNorm/scale -> split four groups
              -> independent softmax/bias/top-k per group
              -> routed experts + identity zero expert
              -> merge/project output + shared SwiGLU expert
```

NVIDIA keeps its existing Triton/CUDA projection, router, top-k, expert and
finalize kernels. Ascend keeps the validated packed BF16 expert weights and the
Prefill/Decode collective schedules. This step does not change either backend's
fusion boundary or communication primitive.

## Shared semantic source of truth

The backend leaves share two device-independent helpers:

- grouped top-k computes probabilities from unmodified FP32 logits, adds the
  correction bias only for selection, optionally renormalizes the selected
  probabilities, and finally applies `routed_scaling_factor`;
- group-aware expert ownership assigns every EP rank an equal contiguous slice
  from every group.

For `G=4`, `E=384` and `EP=8`, rank `r` owns source IDs
`g * E + r * 48 + [0, 48)` for every group `g`, or 192 experts per rank. The
checkpoint loader and the Ascend execution views consume this exact same tuple.
Zero experts are routing-only identities and are not included in the packed
expert weights or ownership tuple. The shared expert always consumes the
ungrouped input and is added after the routed output projection.

## Physical leaves

| Concern | NVIDIA leaf | Ascend leaf |
| --- | --- | --- |
| Router storage | fused classifier over four hidden slices | four classifier modules |
| Expert IDs | globalized flattened IDs | group-local IDs in four views |
| Expert weights | existing `MoELayer` layout | EP-local packed `w13/w2` |
| Prefill schedule | existing CUDA MoE plan | token all-gather, local experts, token reduce-scatter |
| Decode schedule | existing CUDA MoE plan | local experts, EP all-reduce, dense-TP projection |
| Projection/finalize | existing Triton/CUDA fusion | existing Torch/registry/Weight-NZ path |

The selection is resolved once from platform capability while modules are
constructed. There is no per-token device check. The temporary Lite entry also
uses the same Ascend leaf, so deleting that entry in Step 5 will not change the
NPU parameter layout or execution implementation.

## Forward and loader contract

Both leaves accept the common decoder operands
`(hidden_states, num_global_tokens, max_num_tokens_per_gpu, ctx)`. The NVIDIA
leaf ignores `ctx`; the Ascend leaf uses `ctx.forward_mode` to choose its
Prefill or Decode schedule and `ctx.global_num_tokens` for Prefill token
collectives. Architecture names, `world_size == 8`, and `cp_size == 8` do not
select the schedule.

The shared FLASHLocal checkpoint loader is target-driven:

- ordinary CUDA expert layout keeps the existing contiguous loader plan;
- the packed Ascend target exposes its group-aware source-ID tuple, which is
  passed to `MoECheckpointLoader.local_expert_ids`;
- per-group router tensors are loaded through a leaf-owned physical loader
  method, avoiding model-level knowledge of fused versus separate storage;
- separate Ascend shared-expert projections use their target parameter loaders,
  while the existing CUDA fused gate/up loader remains unchanged.

Missing, duplicate, malformed, or inconsistent ownership still fails closed.

## Deliberately unchanged

- CUDA Grouped MoE kernels, expert layout, stream overlap and finalize path;
- Ascend Native MoE plan, packed expert transformation, P/D collectives and
  Weight-NZ admission;
- OE placement/state, temporary strict whole-checkpoint validation, and the
  second model/config entry;
- official Ascend dispatch/combine admission, which remains rejected for the
  current CANN/A2 Lite EP8 shape.

## Validation

The focused portable suite must prove:

- shared routing semantics for bias-only selection, scaling, zero IDs and
  optional renormalization;
- the EP ownership tuple is a bijection and gives 48 experts from every group
  on every EP8 rank;
- both model entries construct the same packed Ascend MoE leaf when that
  capability is selected;
- the shared loader and temporary strict loader place the same group-aware
  expert source IDs into the same packed local rows;
- Prefill/Decode selection follows `ForwardContext`, and invalid/missing
  collective metadata fails closed;
- existing CUDA construction/top-k/identity tests remain green without changing
  the CUDA physical leaf.

The Ascend board suite runs the production packed leaf on NPU0 and retains the
existing EP8 test for all-rank Prefill/Decode collective and numerical parity.
A real GPU gate is required only if the CUDA physical path changes; otherwise
the existing CUDA contract tests are the Step 3 guard when the configured GPU
host is unavailable.

## Validation results

- NPU0 cumulative Lite/runtime suite: `143 passed, 3 skipped`;
- shared routing, construction and target-owned loader subset: `28 passed`;
- portable activation facade regression: `2 passed`;
- production EP8 packed leaf: eight ranks passed Prefill, Decode and Decode
  graph replay against the distributed oracle;
- repository-wide `pre-commit run --all-files`: passed;
- the configured GPU host remained unreachable (`No route to host`), so no real
  NVIDIA execution result is claimed. The existing CUDA kernels, communication
  schedule and output ID dtype remain unchanged.
