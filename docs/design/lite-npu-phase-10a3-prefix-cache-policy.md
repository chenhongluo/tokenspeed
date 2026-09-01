# Lite NPU Phase 10A.3: Prefill Prefix-Cache Policy

## Problem

TokenSpeed enables prefix caching by default. Lite's Prefill model calls the
ordinary MLA extend path, but a cache hit requires the model-specific chunked
prefix replay path. Lite does not implement that replay, so a repeated long
prompt fails at request time with `NotImplementedError` instead of serving the
request.

GSM8K 4-shot evaluation exposes this reliably because adjacent samples share a
long few-shot prefix.

## Decision

The supported 8P8D Lite launcher will pass `--no-enable-prefix-caching` to the
Prefill worker only.

- Prefill will not reuse a prefix computed by another request. A prompt longer
  than `chunked_prefill_size` still has a request-local prefix on its second
  chunk; that continuation is a separate capability.
- Decode remains unchanged. It consumes the transferred request state and does
  not execute the failing Prefill prefix replay.
- KDA/MLA request-local caches, PD transfer, Decode graph, and overlap policy are
  unaffected.

This is the smallest fail-safe configuration for cross-request reuse. It does
not replace request-local chunk continuation, which is handled separately by
the absorbed MLA Extend path in Phase 10A.4.

## Validation

1. Launcher unit tests assert that the Prefill command contains
   `--no-enable-prefix-caching` and the Decode command does not.
2. `bash -n` and the focused launcher test suite must pass.
3. Exact-source 8P8D admission sends two requests with a shared long prefix and
   verifies that both complete without `forward_extend` or scheduler errors.
4. The same service runs EvalScope GSM8K first100 with the frozen 4-shot,
   batch-2, greedy configuration; all 100 predictions must be present and free
   of transport/runtime errors.

## Deferred optimization

Re-enable cross-request Prefill prefix caching only after Lite has a numerical
oracle and an NPU test for replay covering hybrid KDA/MLA/OE state publication.
