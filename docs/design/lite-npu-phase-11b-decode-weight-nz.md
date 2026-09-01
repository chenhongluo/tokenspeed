# Lite NPU Phase 11B: Decode-only Weight-NZ production design

## 1. Scope

Phase 11B prepares a bounded set of Lite BF16 Decode GEMM weights in Ascend FRACTAL_NZ storage
format after checkpoint loading. Prefill remains in ND format. The optimization is opt-in through
one server argument and is enabled only on the Decode command in the supported 8P8D launcher.

This phase does not change model mathematics, checkpoint layout, tensor parallelism, routed-expert
GMM weights, graph mode, overlap scheduling, quantization, OE host residency, or any attention/cache
protocol. It is independent from Phase 11A fused Add-RMSNorm and can be disabled without changing
that optimization.

## 2. Existing implementation boundary

The model loader already invokes `process_weights_after_loading` for every module after all
checkpoint tensors have been loaded. Lite's parent KDA, MLA, and Grouped-MoE modules are visited
before their child `_Weight` modules. Phase 11B reuses that lifecycle:

1. parent modules create their existing derived tensors, including KDA packed convolution weights
   and MLA `w_kc`/`w_vc`;
2. selected child `_Weight` modules prepare their persistent BF16 weight once;
3. Decode forward consumes the prepared weight without conversion, allocation, or host sync.

The vendor operation stays below the unified kernel boundary. Runtime/model code imports
`tokenspeed_kernel`; only `tokenspeed-kernel-npu` imports `torch_npu`. A registry, factory, plan
object, new dependency, or forward-time format branch is unnecessary for this one-time operation.

## 3. Exact whitelist

The target Lite model has 28 layers: 21 KDA and 7 MLA. Dense TP is eight on Decode. The following
table is the complete whitelist for each Decode rank:

| Module weight | Loaded shape | Stored shape | Forward contract | Count |
| --- | ---: | ---: | --- | ---: |
| KDA `o_proj` | `[3072, 512]` | `[3072, 512]` | existing `F.linear(x, w)` | 21 |
| MLA `q_a_proj` | `[1536, 3072]` | `[3072, 1536]` | `torch.matmul(x, w)` | 7 |
| MLA `kv_a_proj_with_mqa` | `[576, 3072]` | `[3072, 576]` | `torch.matmul(x, w)` | 7 |
| MLA `q_b_proj` | `[6144, 1536]` | `[6144, 1536]` | existing unified MLA query projection | 7 |
| MLA `o_proj` | `[3072, 4096]` | `[3072, 4096]` | existing `F.linear(x, w)` | 7 |
| Grouped MoE `proj_output` | `[3072, 384]` | `[3072, 384]` | existing `F.linear(x, w)` | 28 |
| Shared expert `down_proj` | `[3072, 384]` | `[3072, 384]` | existing `F.linear(x, w)` | 28 |

This produces exactly 105 format-29 tensors per Decode rank and zero per Prefill rank. Only MLA
`q_a_proj` and `kv_a_proj_with_mqa` change logical storage orientation. Their two direct projection
calls use the child module's prepared-weight forward contract. All other callers retain their
current `[N,K]` ABI.

The following weights are deliberately excluded:

- KDA q/k/v/output-gate/forget/beta projections and causal-convolution weights;
- MLA output-gate, `kv_b_proj`, and derived `w_kc`/`w_vc` tensors;
- Grouped-MoE input projection, shared gate/up projections, routers, and all routed-expert GMM
  weights;
- embeddings, language-model head, norms, cache/state tensors, and OE host tables/projections;
- every Prefill weight.

Routed-expert weights already have a separate MoE post-load ABI and were admitted without internal
format in Phase 6. They must not be swept into a global cast operation.

## 4. Format preparation contract

The NPU helper accepts a two-dimensional NPU weight and an optional transpose request. It:

1. rejects non-NPU or non-2D input;
2. optionally performs `weight.transpose(0, 1).contiguous()`;
3. saves whether `torch.npu.config.allow_internal_format` existed and its previous value;
4. enables internal format only for the `torch_npu.npu_format_cast(weight, 29)` call;
5. requires `torch_npu.get_npu_format(result) == 29` and raises `RuntimeError` otherwise;
6. restores the previous value, or removes the temporary attribute when it did not previously
   exist.

The helper does not leave internal format globally enabled. It also does not silently return ND
when the requested layout is unavailable. The target wheel was directly tested both with and
without internal format enabled: disabled conversion remained format 2; enabled conversion
produced format 29. A format-29 tensor remained executable after restoring the original runtime
configuration.

The official Torch-NPU native schema publishes both `npu_format_cast(Tensor, int)` and
`get_npu_format(Tensor) -> int`:

<https://gitcode.com/Ascend/pytorch/blob/master/torch_npu/csrc/aten/npu_native_functions.yaml>

The public format-cast documentation is under the official Torch-NPU custom API reference:

<https://gitcode.com/Ascend/op-plugin/tree/26.0.0/docs/zh/custom_APIs/torch_npu>

## 5. Role and flag policy

`ServerArgs` adds `--npu-enable-weight-nz`, defaulting to false. The supported launcher adds a
validated `LITE_DECODE_WEIGHT_NZ=0|1` environment override, defaulting to one, and appends the flag
only to the Decode command. The Prefill command never receives it.

The Lite post-load hook requires all three conditions:

- the selected `_Weight` carries an explicit standard or transposed NZ marker;
- the tensor device is NPU;
- the server argument is enabled and `disaggregation_mode == "decode"`.

CPU, CUDA, Prefill, disabled Decode, and unmarked weights remain unchanged. The launcher override
provides an exact off/on service A/B without adding a forward branch.

## 6. Direct-board evidence

The target environment is `torch_npu 2.9.0.post2` on Atlas 910B. All candidate shapes were tested
with BF16 input and weight, 20 warmups, nine rounds of 2,000 NPU-event-timed calls, and token counts
one and two. Every output was finite and agreed with the corresponding ND reference under
`rtol=atol=1e-2`.

Median microseconds per call for the production layout candidates were:

| Weight shape | Tokens | Current ND | Selected NZ | Relative change |
| --- | ---: | ---: | ---: | ---: |
| KDA o `[3072,512]` | 1 | 8.615 | 8.546 | 0.8% faster |
| KDA o `[3072,512]` | 2 | 8.664 | 8.589 | 0.9% faster |
| MoE output/down `[3072,384]` | 1 | 8.551 | 8.485 | 0.8% faster |
| MoE output/down `[3072,384]` | 2 | 8.483 | 8.383 | 1.2% faster |
| MLA q-a `[1536,3072]` | 1 | 8.377 | 6.874 | 17.9% faster |
| MLA q-a `[1536,3072]` | 2 | 8.482 | 6.825 | 19.5% faster |
| MLA kv-a `[576,3072]` | 1 | 8.487 | 6.764 | 20.3% faster |
| MLA kv-a `[576,3072]` | 2 | 8.668 | 6.840 | 21.1% faster |
| MLA q-b `[6144,1536]` | 1 | 9.021 | 8.926 | 1.1% faster |
| MLA q-b `[6144,1536]` | 2 | 9.028 | 8.903 | 1.4% faster |
| MLA o `[3072,4096]` | 1 | 11.336 | 11.247 | 0.8% faster |
| MLA o `[3072,4096]` | 2 | 11.166 | 10.984 | 1.6% faster |

For q-a and kv-a, "Selected NZ" means transposed format-29 storage plus `matmul`; for all other
rows it means format-29 storage in the existing `[N,K]` layout. Transposing q-b or MLA o was slower
than their current caller contract and was rejected.

An NPUGraph board covered all six distinct call shapes at BS1 and BS2. Capture, input mutation, and
replay passed in 12/12 cases; every stored weight reported format 29, all outputs were finite, and
the largest absolute difference from the reference was `0.00390625`. These leaf results justify
admission, not an end-to-end performance claim.

## 7. Implementation and validation

The implementation commit must contain the smallest production change:

- one unified `prepare_weight_nz(weight, transpose=False)` kernel entry;
- one `_Weight` post-load mode and forward contract;
- markers only at the seven whitelist construction sites;
- two MLA direct projection call-site changes;
- one server argument/global propagation entry and one Decode-only launcher option.

Focused validation must prove:

- parser/global-argument propagation and launcher Decode-only default/off override;
- role/device/disabled no-op behavior and exact whitelist counts;
- expected transposed and standard shapes, format 29, and fail-closed format mismatch;
- CPU behavior remains ND and mathematically unchanged;
- NPU eager and NPUGraph BS1/BS2 precision with changed replay inputs;
- cumulative Lite focused tests and full pre-commit.

Final validation uses the exact implementation commit and the same 8P8D topology and workload on
both sides of the A/B. It must include a four-token smoke, a 1,100-token Prefill continuation,
late-admission BS2, finite/fatal log scans, Decode graph BS1/BS2, and GSM8K first 100 with the frozen
Phase 10/11A EvalScope configuration. Service timing is reported separately from leaf timing.

## 8. Rollback

Set `LITE_DECODE_WEIGHT_NZ=0` to retain canonical ND checkpoint weights without changing graph,
overlap, or Phase 11A. The implementation is retained only if exact-source service validation has
no new startup failure, non-finite value, graph fallback, attributable GSM8K regression, or material
end-to-end slowdown. A failed format request stops model loading rather than silently running a
different layout.
