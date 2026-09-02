# Flash-Lite GPU/NPU runtime convergence: Step 0 behavior guards

## Scope

This step freezes the behavior of the two temporary model entries before they are
merged. It is based on `longcat-lc/lite-main@96d39378d402397ca01ac8b1ee685e9e7e75595c`.
There is no production-code change in this step.

The contract is deliberately split into two parts:

- model semantics that must already agree and remain shared after convergence;
- physical layout and state-provider choices that may differ by resolved capability.

## Shared semantic contract

The same synthetic checkpoint config must produce the same values for:

- layer count and the KDA/MLA layer pattern;
- KDA heads, dimensions, convolution width, full-rank gate and NoPE mode;
- MLA low-rank dimensions, scale switches and output gate;
- Grouped MoE group count, routed/shared/zero experts, top-k and normalization;
- OE branch count, table-size base, special-token policy and normalization.

`test_dual_entries_derive_the_same_model_semantics` compares these values directly.
It intentionally does not compare the overloaded `num_experts` alias: the current GPU
entry uses it for experts per group, while the NPU entry uses it for all groups. The
canonical fields are `n_routed_experts` and `moe_group_size`.

## Allowed physical differences

Both entries consume the same checkpoint source keys, but some keys currently land in
different runtime parameters:

| Module | GPU entry | NPU entry |
| --- | --- | --- |
| KDA input projection | separate `q/k/v/g/fA/bA` parameters | one packed projection parameter |
| Grouped MoE experts | standard per-expert logical parameters | EP-local packed `w13/w2` parameters |
| OE table | device-sharded table | host-resident full table |
| OE state | `RuntimeStates` full history | cache-arena checkpointed tail |

`test_checkpoint_sources_preserve_current_physical_layouts` locks representative
KDA, MLA, MoE and OE source-key mappings without requiring either physical layout to
become the other.

## Reused numerical guards

The existing tests remain the numerical source of truth:

| Area | Existing guard |
| --- | --- |
| GPU config/router/OE policy/checkpoint aliases | `test/runtime/models/test_flash_kda.py` |
| NPU config, strict loader and packed shapes | `test/runtime/test_lite_model_loader.py` |
| KDA projection, Prefill/Decode recurrence and output gate | `test/runtime/test_lite_kda_eager.py` |
| MLA Prefill/Decode, cache update and output gate | `test/runtime/test_lite_mla_eager.py` |
| Grouped MoE routing, zero/shared experts and P/D collectives | `test/runtime/test_lite_grouped_moe.py` |
| Host OE math, checkpointed tail and slot reuse | `test/runtime/test_lite_oe.py` |
| Device OE full-history ownership and graph view | `test/runtime/execution/test_runtime_states.py`, `test/runtime/execution/test_request_token_history_graph.py` |

Step 0 adds no duplicate oracle. Later steps must keep the relevant rows green and may
update only the explicit physical-layout expectations when ownership is intentionally
moved behind a shared leaf or capability resolver.

## Validation

Run the focused CPU/import suite in an environment containing the repository's
TokenSpeed kernel and scheduler packages:

```bash
PYTHONPATH=python:tokenspeed-kernel/python:tokenspeed-kernel-npu/python \
python -m pytest -q \
  test/runtime/test_lite_dual_entry_contract.py \
  test/runtime/models/test_flash_kda.py \
  test/runtime/test_lite_model_loader.py \
  test/runtime/test_lite_kda_eager.py \
  test/runtime/test_lite_mla_eager.py \
  test/runtime/test_lite_grouped_moe.py \
  test/runtime/test_lite_oe.py \
  test/runtime/execution/test_input_buffer_request_token_history.py \
  test/runtime/execution/test_request_token_history_graph.py \
  test/runtime/execution/test_runtime_states.py \
  test/runtime/test_request_history_seed.py
```

Accelerator-only cases remain part of their existing GPU/NPU board suites; Step 0 does
not add a service launch or change the two-card GPU / 8P8D NPU acceptance matrix.

The complete focused suite passed on one Ascend 910B device: `106 passed`. The two
reported warnings were environment/backend warnings; there were no failures, skips,
NaN/Inf reports or production-code changes.
