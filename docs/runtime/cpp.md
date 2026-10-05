# C++ runtime (v0.10)

Production evaluator + engine integration + SIMD + low latency.

## Layout

- `core/kernels/ref_kernels.*` — scalar reference, exact semantics.
  Used for validation and as fallback.
- `core/kernels/simd_kernels.*` — dispatched path. AVX2 `matVec`
  where compiled with AVX2, otherwise delegates to ref. Runtime
  detection via `__builtin_cpu_supports`, `setPathForTest` for
  deterministic scalar-vs-SIMD comparison.
- `core/simd/simd.h` — thin dispatcher, kept for existing call
  sites. All mixer/head code goes through it.
- `core/inference/evaluator.h/.cpp` — thin glue, no math.
- `core/inference/dump.*` — deterministic intermediate dump
  (`--dump-intermediate` equivalent via `rune_diff --dump-dir`).
- `core/model_io/` — format 1+2 loader, canonical + legacy field
  names, feature/quant/arch validation, size guards, hash verify
  for format 2 and all dense/adaptive files. `model_error.*` is
  the structured error model (no silent fallback).
- Tools: `build/rune_eval` (single eval), `build/rune_diff`
  (self-check + dumps), `build/rune_bench_v10` (per-stage + MT).

## Threading

Evaluator holds no global mutable state. One `Evaluator` per
thread sharing immutable tables/model. `rune_bench_v10 --threads N`
measures scaling; linear scaling is never assumed.

## Unsupported ISA

Missing AVX2 falls back to scalar silently but visibly
(`path` is printed in every bench/diff line). No crash on
unsupported instructions.
