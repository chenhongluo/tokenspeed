# Lite NPU Phase 10A.4: MLA Prefill Continuation

## Problem

Disabling Prefill prefix caching prevents reuse across requests, but it does not
remove a prefix created by an earlier chunk of the same request. With
`chunked_prefill_size=1024`, the second chunk has `extend_prefix_lens > 0`.

Lite already selects absorbed MLA Extend when the backend publishes
`use_absorbed_cached_extend=true`. On Ascend, the registered Extend kernel only
declares page size 128, while the bounded Lite service uses kernel page size 64.
The capability query therefore returns false and ordinary explicit Prefill
fails loud rather than reading the existing latent pages.

## Existing data flow

No model-specific replay loop is required:

1. `MLAAttnBackend` checks the registered absorbed-Extend capability for the
   live page size and query shape.
2. Its Prefill metadata publishes `use_absorbed_cached_extend=true`, the full
   page table, cumulative query lengths, and cumulative visible KV lengths.
3. `LiteMLAParameters` writes the current chunk's compressed latent to its live
   cache locations and builds the absorbed query with the existing `W_KC`.
4. `MLAAttnBackend.forward_extend` calls the existing
   `mla_extend_with_kvcache` facade over history plus current chunk.
5. The existing value projection, output gate, and `o_proj` complete the layer.

KDA recurrent/conv state and OE context continue through their existing
request-local cache contracts. Decode, PD transfer, graph, and cross-request
prefix policy are unchanged.

## Decision

Extend the Ascend `mla_extend_with_kvcache` registration from page size 128 to
page sizes 64 and 128. Do not add a Lite replay implementation, a launcher
fallback, or a larger Prefill chunk.

This is a capability correction: the existing Torch-NPU leaf accepts page size
64. A 910B board with Lite shape H32/R512, prefixes `[1024,127]`, and query
lengths `[64,17]` completed with finite output, output relative L2
`0.00186938`, and LSE maximum absolute error `1.43e-6` against a FP32 oracle.

## Validation

1. A CPU model-path test proves cached Extend selects absorbed attention and
   still writes the current latent before attention.
2. An Ascend test asserts the public capability query admits page size 64 and
   runs the page-64 leaf across a page boundary against the FP32 oracle.
3. Exact-source 8P8D serves a prompt longer than 1024 tokens with cross-request
   prefix caching disabled and no `forward_extend` error.
4. EvalScope GSM8K first100 completes in Decode eager and Decode graph modes
   with identical evaluation settings and no missing/error responses.

## Rollback

Remove page size 64 from the single Ascend kernel registration. No model,
backend, cache, scheduler, or launcher state must be reverted.
