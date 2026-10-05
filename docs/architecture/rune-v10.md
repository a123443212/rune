# RUNE v0.10 Architecture: Dual Runtime + Shared Spec

Question: can C++ and Rust be two independent implementations of
one RUNE specification, loading the same `.rune`, producing the
same result, benchmarked transparently.

## Three layers

```text
Layer 1 — RUNE Specification   spec/
Layer 2 — Runtime Kernels       core/kernels/*  +  crates/rune-kernel
Layer 3 — Application           engine/CLI/bench/diff
```

Layer 1 is canonical. Layer 2 has two implementations. Layer 3
differs by language on purpose (C++ engine, Rust CLI/research).

## What changed from v0.9

- `spec/` is new and authoritative: features, tensors, quant,
  architecture, model format, numerical contract, versions,
  test vectors. Nothing in v0.9 had this.
- `.rune` is now format 2 with `architecture_id`,
  `architecture_version`, `feature_version`, `tensor_metadata`,
  `quantization_metadata`, `model_hash` for every model. Format 1
  still loads. One file loads in Python, C++, Rust with the same
  hash.
- C++ kernels split into `ref` (scalar, exact) and `kern`
  (dispatched, AVX2 where compiled, scalar fallback). `simd.h`
  is now a dispatcher, not a second implementation.
- Rust is now a real runtime: `crates/rune-spec/model/kernel/
  runtime/data/cli`. Board with FEN + UCI make/unmake, features,
  accumulator (fp32 + int paths), mixer, head, evaluator,
  model loader, CLI (`eval/inspect/bench/diff/model-info`).
  `rust-data/` stays as the data plane and is not rewritten.
- Golden vectors in `spec/test-vectors/v10/` are the only
  cross-language oracle. Generator is `tools/golden/`.
- Differential testing is `tools/diff/cross_check.py` (Python↔C++
  ↔Rust triangle) plus `build/rune_diff` (C++ stage self-check)
  and per-language vector tests. Tolerances are per-stage, never
  one global epsilon.
- Benchmarks are `tools/bench/run_bench.py` over
  `benchmark/positions/` (quiet/tactical/endgame/king_attack/
  random), reporting per-stage us, eval/sec, threads, and the
  scalar-vs-SIMD split per language.

## Roles after v0.10

```text
C++:  production engine runtime, primary low-latency path
Rust: secondary runtime, research runtime, data infra,
      cross-validation platform
Python: training + reference numerical path, never production
```

No FFI between C++ and Rust in v0.10. The boundary is the
`.rune` artifact and the shared vectors.
