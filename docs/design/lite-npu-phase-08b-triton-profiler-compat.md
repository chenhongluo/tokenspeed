# Lite NPU Triton profiler compatibility

## Problem

The Ascend Triton distribution provides the compiler and runtime used by NPU
kernels, but some compatible builds omit the optional `triton.profiler` module
and the CUDA-only `tl.extra.cuda` namespace. The NPU adapter imported the
profiler unconditionally and assumed that namespace existed, so model probes
failed before loading a checkpoint even when profiling and CUDA PDL were
disabled.

## Contract

- Triton compilation and runtime imports remain mandatory.
- Proton profiling is optional, matching the common kernel profiling layer.
- When `triton.profiler` is absent, the adapter exposes `proton = None`; the
  common profiling API already turns profiling calls into no-ops with a
  warning.
- When `tl.extra.cuda` is absent, the adapter creates only the namespace needed
  for the existing disabled PDL no-ops; it does not enable a CUDA execution
  path.
- No model, kernel selection, numerical path, or NPU execution behavior changes
  when profiling is disabled.

## Validation

The focused unit test imports the adapter against a minimal Triton module that
has language/runtime symbols but neither a profiler nor a CUDA namespace. It
verifies that import succeeds with `proton = None` and both disabled PDL no-ops
are installed. Exact-source NPU validation must additionally import the full
kernel package before checkpoint loading.
