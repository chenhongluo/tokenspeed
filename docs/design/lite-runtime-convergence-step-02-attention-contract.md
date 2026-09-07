# Flash-Lite GPU/NPU runtime convergence: Step 2 attention contract

## Scope

This step gives Lite KDA and MLA one model-level contract while retaining the
two already validated physical implementations:

- NVIDIA keeps the separate KDA projections and fused Kimi MLA projection;
- Ascend keeps the packed KDA projection and separate Weight-NZ MLA projections;
- both paths use the same KDA backend ABI and the same DeepSeek/Kimi MLA cache,
  Prefill, absorbed-Decode and output-gate dataflow.

Grouped MoE, OE placement/state, the temporary `LiteForCausalLM` entry and its
strict source-layout probe remain until later convergence steps.

## Parallel ownership

Attention modules use the three mappings established in Step 1:

| Concern | Mapping |
| --- | --- |
| MLA parameter/head geometry | `mapping.mla_weight` |
| MLA execution, cache and communication | `mapping.attn` |
| KDA parameter, state and communication | `mapping.linear_attn` |

`DeepseekV3AttentionMLA` accepts an optional component mapping and defaults to
`mapping.attn`, preserving every existing caller. Lite passes
`mapping.mla_weight`, which is TP1 for both 8P8D roles and for the bounded TP8
execution topology.

## KDA physical leaves

The common KDA forward contract is:

```text
(positions, hidden_states, ctx, out_cache_loc, comm_manager) -> hidden partial
```

Both leaves pass the same semantic operands to `ctx.attn_backend.forward` and
return an output-projection partial. Their parameter layouts intentionally
differ:

| Leaf | Projection layout | Initial admission |
| --- | --- | --- |
| `SeperateFLASHLocal` | separate q/k/v/g/fA/bA | NVIDIA |
| `PackedFLASHLocalKDA` | one q/k/v/g/fA/bA parameter | Ascend |

The packed leaf lives outside `models/lite.py`. Its packed parameter owns the
component loader, offsets, shape/dtype validation and component IDs. The strict
temporary loader canonicalizes source keys and invokes that parameter loader;
it does not inspect the concrete KDA class.

The packed layout is selected once during model construction. It does not add a
per-token device branch and does not change backend fusion selection.

## MLA shared dataflow

`KimiLinearMLAAttention` remains the semantic owner of:

- NoPE query/latent projection;
- q/kv normalization and absorbed weights;
- model-owned MLA cache writes;
- explicit Prefill and absorbed Decode;
- output gate and output projection.

`SeparateProjectionKimiLinearMLAAttention` overrides only the input projection
stage and physical linears needed by the Ascend checkpoint path. It inherits the
entire attention/cache forward from `KimiLinearMLAAttention`. Its q-a/kv-a/q-b/o
weights retain the existing optional Weight-NZ preparation through a local
Linear leaf whose post-load hook calls the public `tokenspeed-kernel` boundary.

The vendor-neutral activation facade owns sigmoid-times-value. CUDA/ROCm retains
the existing Triton implementation; Ascend and CPU use the equivalent in-place
Torch fallback. The Kimi model no longer imports this epilogue from a Triton
module directly.

## Construction and loader rules

- the shared FLASHLocal decoder resolves the physical attention leaf once;
- automatic resolution keeps current NVIDIA and Ascend choices;
- the temporary Lite decoder instantiates the same Ascend leaves, so deleting it
  in Step 5 will not change attention implementation;
- source checkpoint names remain unchanged;
- the temporary strict loader drives packed target assembly through the target
  parameter's loader; moving that source-layout contract into the single model
  entry remains part of Step 5;
- missing, duplicate or malformed packed components fail closed.

## Deliberately unchanged

- NVIDIA KDA weights, projection calls, beta ABI and fused epilogue;
- NVIDIA MLA fused q/kv/g layout and optimized projection selection;
- Ascend KDA/MLA kernel registrations and graph behavior;
- Grouped MoE communication and fused kernels;
- OE host/device placement and context-state providers;
- temporary architecture/config resolution.

## Validation

The focused CPU/meta tests must prove:

- existing callers retain `mapping.attn` MLA geometry by default;
- Lite MLA uses full replicated heads from `mapping.mla_weight` even when
  attention execution is TP8;
- both physical KDA leaves keep their parameter trees and source mappings;
- the packed parameter loader validates and assembles all six components;
- the Ascend MLA leaf inherits the shared Kimi/DeepSeek cache/dataflow and
  matches the existing Lite mathematical oracle;
- CPU sigmoid fallback is in-place and numerically equivalent;
- Step 0/1 contracts remain green.

The Ascend focused suite additionally runs the production packed KDA and shared
MLA leaves on NPU0 for Prefill and Decode. Because the NVIDIA numerical path is
not edited, GPU validation is limited to construction/loader regression unless
a reachable GPU host is available; any change that touches its physical leaf
would require the full two-GPU service gate.

## Validation results

- NPU0 cumulative Lite runtime suite: `145 passed, 4 skipped`;
- portable sigmoid fallback contract: `2 passed`;
- shared MLA Prefill/Decode leaf: `12 passed`, including live cache-write order,
  output gate and real Ascend execution;
- separate KDA mapping and shared/default MLA geometry: `53 passed` in the
  focused construction/loader/mapping subset;
- repository-wide `pre-commit run --all-files`: passed;
- the configured two-GPU host was unreachable (`No route to host`), so this
  step did not claim a real NVIDIA execution gate. The NVIDIA projection and
  kernel implementations are unchanged; platform-neutral tests lock their
  separate layout and `mapping.linear_attn` geometry.
