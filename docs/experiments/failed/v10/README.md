# Failed experiments (v0.10)

Kept readable, never deleted. Each entry: what was tried, what
was measured, why it stays out.

## Rust SIMD in v0.10 — DONE, graduated to results

Was: dispatch plumbing only. Now: `crates/rune-kernel/src/simd.rs`
ships an AVX2+FMA `mat_vec` lane, 1:1 port of C++ `matVecAvx2`,
`unsafe` contained in the kernel behind a safe wrapper with
size guards and scalar fallback. Evidence in
`docs/performance/rune-v10-benchmarks.md`: lanes bit-identical
C++↔Rust, ~4x eval speedup both languages.

## Dynamic dispatch for kernels — rejected

Considered a trait-object kernel set for scalar/SIMD switching.
Rejected: branch cost unmeasured but API complexity certain;
static dispatch + `setPathForTest` covers validation with less
surface. Revisit only with profiles showing dispatch overhead
matters.

## Single global epsilon — rejected

A single 1e-5 for the whole net hid a head-matrix transpose bug
during development (caught only after per-stage split). Policy
is now per-stage exact/tolerant in `spec/numerical-contract.md`.

## FFI C++↔Rust in v0.10 — deferred

Prototype `rune_eval_*` C ABI sketched but not shipped. No
consumer needed it; artifact boundary (`.rune` + vectors) was
enough for validation and benchmarking. Revisit for embedding.

## Dense int8 claim — DONE, graduated to results

Was: unmeasured. Now: `dense-b-int8.rune` + `dense-b-int16.rune`
shipped by `tools/golden/gen_models.py`; Python↔C++ parity via
`tools/diff/dense_check.py` (9/9 positions, fp32↔int8 gap ~7e-06);
C++ `testDenseQuantCloseness` and pytest `test_v10_debts.py`
lock it in. Rust loads dense fixtures (`model-info`) but its
evaluator rejects RUNE-03-*/RUNE-04/RUNE-05 by allowlist — see
"Rust silent mis-eval of RUNE-04" below, fixed.

## Rust silent mis-eval of RUNE-04 — found and fixed

`Evaluator::from_model` accepted any arch whose tensors happened
to match the ATTN/GAB names. The adaptive fixture exports
`wq/bq/.../w1/...`, so Rust built a wrong-but-plausible evaluator
(value=-0.046485, meaningless: refinement weights run as a plain
attention net, no cheap path, no routing). Fixed with an explicit
allowlist (RUNE-SFNN/MLP/ATTN/ATTN-GAB/REL-02); everything else
fails closed with `UnsupportedArch`. Regression test:
`unsupported_arch_fails_closed`. Lesson for v1.0: name-matching
is not a contract; the allowlist is now part of the Rust runtime
spec in `docs/runtime/rust.md`.
