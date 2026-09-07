# Flash-Lite GPU/NPU runtime convergence: Step 1 parallel seams

## Scope

This step makes the existing attention parallel domains explicit before either
temporary Lite model entry is changed. It does not change model math, checkpoint
parameter layout, backend selection, fusion boundaries, OE placement or state
ownership.

## Mapping contract

One decoder can use three distinct attention-related mappings:

| Mapping | Owner | Meaning |
| --- | --- | --- |
| `mapping.attn` | full attention | attention execution and cache distribution |
| `mapping.linear_attn` | KDA | KDA parameter, recurrent-state and execution TP |
| `mapping.mla_weight` | MLA | MLA parameter/component TP only |

`linear_attn` now reuses `AttentionLayerMapping` with `cp_size=1`. This preserves
its existing TP/DP groups while giving it the same `has_tp`, `cp_size` and
`scatter_index()` contract needed by the shared communication manager.

`--mla-weight-tp-size` defaults to the previously declared attention TP. The Lite
8P8D launcher explicitly sets it to `1`: every rank keeps complete MLA components,
while Prefill CP8 and Decode DP8 remain properties of `mapping.attn`. No model name,
architecture string or world-size heuristic participates in this choice.

## Layer-specific communication

`CommManager` accepts an optional `attn_mapping`. Existing callers omit it and keep
the prior `mapping.attn` behavior. The later shared decoder can pass
`mapping.linear_attn` for KDA layers and `mapping.attn` for MLA layers without
duplicating residual, all-reduce or reduce-scatter control flow.

The mapping changes only token partition/group selection. The backend collective
implementations and fused all-reduce-plus-norm kernels are unchanged.

## Cache geometry

The Kimi-K3 hybrid cache recipe always obtains KDA state TP from
`mapping.linear_attn`. Its default equals attention TP, so existing Kimi-K3
deployments keep their geometry; Lite CP8/DP8 deployments can retain KDA TP8
without an architecture special case.

`MLAConfig` obtains component geometry from `mapping.mla_weight`. Attention
execution and cache ownership continue to use `mapping.attn` outside that config.

## Deliberately unchanged

- GPU separate and NPU packed KDA parameters and kernels;
- both temporary model entries and their loaders;
- RuntimeStates full-history and Host/checkpointed-tail external inputs;
- graph, prefix-cache and speculative admission gates;
- Grouped MoE and OE implementation.

The architecture gates stay until Step 4 has an explicit placement/state capability
to replace them.

## Validation

The focused tests cover:

- default and explicit MLA weight TP resolution, including CP8 and DP8 roles;
- KDA mapping's shared attention interface and layer-specific `CommManager` token
  partitioning;
- KDA cache shape independence from `model_type`;
- explicit MLA TP1 geometry independent from the architecture string;
- the 8P8D launcher's explicit `--mla-weight-tp-size 1` contract;
- the unchanged Step 0 dual-entry and Lite numerical behavior guards.

Run the focused suite with the repository's runtime/kernel packages available:

```bash
PYTHONPATH=python:tokenspeed-kernel/python:tokenspeed-kernel-npu/python \
python -m pytest -q \
  test/runtime/test_linear_attn_mapping.py \
  test/runtime/test_lite_hybrid_cache.py \
  test/runtime/test_kimi_k3_cache_pool.py \
  test/runtime/test_lite_dual_entry_contract.py \
  test/runtime/test_lite_model_loader.py \
  test/runtime/test_lite_kda_eager.py \
  test/runtime/test_lite_mla_eager.py \
  test/runtime/test_lite_grouped_moe.py \
  test/runtime/test_lite_oe.py \
  test/runtime/execution/test_input_buffer_request_token_history.py \
  test/runtime/execution/test_request_token_history_graph.py \
  test/runtime/execution/test_runtime_states.py \
  test/runtime/test_request_history_seed.py \
  test/ci_system/test_lite_npu_pd_launcher.py \
  test/ci_system/test_lite_role_checkpoint_probe.py
```

The cumulative suite passed with `154 passed, 1 skipped` on Ascend NPU0. The
single skip is the existing CUDA-only Kimi-K3 physical-pool test; its cache
geometry contract and all Lite CPU/meta/NPU numerical guards passed.
