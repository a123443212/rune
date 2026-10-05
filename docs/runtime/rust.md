# Rust runtime (v0.10)

Secondary + research runtime, real evaluator, not a data utility.

## Layout

```text
crates/rune-spec    constants, versions, hash, tolerances
crates/rune-model   .rune loader, validation, hash, errors
crates/rune-kernel  scalar reference (lib.rs) + AVX2 lane
                    (simd.rs), dispatched via mat_vec
crates/rune-runtime board (FEN+UCI make/unmake), features,
                    accumulator, mixer, head, evaluator,
                    thread-local contexts, dump
crates/rune-data    dataset plane (parse/filter/shard/sample),
                    shares board/features with runtime but owns
                    no evaluator state
crates/rune-cli     rune eval/inspect/bench/diff/model-info
```

Legacy `rust-data/` workspace is untouched and still builds;
new `crates/rune-data` is the v0.10 data-plane contract. They
agree on feature semantics and are tested together.

## Design choices

- Reference first, then SIMD: scalar kernels passed diff before
  any intrinsic landed. The AVX2+FMA `mat_vec` lane
  (`simd.rs`) is a 1:1 port of C++ `matVecAvx2`, bit-identical
  to it on all 8 fuzz shapes, within contract tol of scalar.
  `mat_vec` dispatches (auto-detect + `set_path_for_test`
  override mirroring C++); `mat_vec_scalar` stays the exact
  reference golden tests pin.
- Unsafe policy: the single `unsafe` block lives in
  `simd.rs::mat_vec_avx2_unsafe`, calling only
  `std::arch::x86_64` intrinsics under
  `#[target_feature(enable = "avx2,fma")]`. The safe wrapper
  checks slice lengths (checked `rows*cols`, bias presence) and
  falls back to scalar when AVX2/FMA is absent — no crash on
  unsupported ISA, no UB on bad shapes (panics like scalar).
  Justification: ~4x eval speedup, measured in
  `docs/performance/rune-v10-benchmarks.md`.
- Supported architectures (allowlist, fail-closed): the
  evaluator implements RUNE-SFNN/MLP/ATTN/ATTN-GAB/REL-02 only.
  RUNE-03-*/RUNE-04/RUNE-05 load fine in `rune-model`
  (`model-info` works) but `Evaluator::from_model` rejects them
  with `UnsupportedArch`. Name-matching is not a contract: an
  earlier build silently ran adaptive refinement weights as a
  plain attention net. Regression test
  `unsupported_arch_fails_closed`.
- Ownership: model is immutable and shared (`&RuneModel`
  arrays copied once into `Tables`); evaluator state
  (`Accumulator` + feature list) is per-instance, `Send`-safe
  by construction, no global mutable state.
- Errors: `Result<T, RuntimeError>` everywhere. Malformed
  models fail closed; fuzzed loader inputs never panic on
  allocation (size guards before `Vec` growth).
- Board scope: FEN parse + UCI apply/unmake + pseudo-move
  count is enough for the evaluation path. Full legal movegen
  for search stays a C++ job; Rust does not pretend to be a
  full engine in v0.10.
