# RUNE v0.10 — v0.9 Audit (input to Dual Runtime)

Scope: everything v0.9 leaves behind that v0.10 must either keep,
freeze into shared spec, reimplement, or remove. Criterion is
narrow: does this component describe mathematical semantics that
C++ and Rust must agree on, or is it tooling around that core.

Inspected 2026-10-05 against git HEAD: `core/` (board, features,
accumulators, architectures, inference, model_io, simd, position),
`training/` (features, models, export, quantize), `rust-data/`
(board, features, format, record, pipeline, hash), `benchmarks/`,
`bindings/`, `tests/`, `tools/`, `configs/v09/`, docs v09 chain.

## 1. Component table

| Component | Location | Verdict | Reason |
| --------- | -------- | ------- | ------ |
| Grouped feature extraction (8 groups, vocab [256,256,256,128,128,512,512,64], threats+mobility+global) | `core/features/feature_set.*`, `training/features/python_features.py`, `rust-data/.../features.rs` | MOVE TO SHARED SPEC + REIMPLEMENT IN RUST (runtime) | Logic already triple-mirrored and matching on startpos, but canonical source is implicit. Must freeze as `spec/features/` with version `grouped_hkav2_fullthreats_v01` and make C++/Python/Rust all consume generated vectors. Rust copy today lives in data crate, needs move to runtime kernel |
| Board model + move gen + make/unmake | `core/board/*`, `core/position/*`, `rust-data/.../board.rs` | KEEP (C++) + REIMPLEMENT IN RUST (runtime board) | C++ board is production path. Rust board only parses FEN for dataset, no move make/unmake, no legal movegen. Runtime parity (make/unmake differential) needs a real Rust board |
| GroupedAccumulator fp32 refresh/applyDiff/clip01 tokens | `core/accumulators/grouped_accumulator.*` | MOVE TO SHARED SPEC + REIMPLEMENT IN RUST | Semantics are simple (sum per group, clip01) but layout (row-major 8x32, float32 acc) is nowhere specified. Canonical numerical contract needed |
| FlexAccumulator + TokenLayout (tokens 6/8/10, dim 24/32/40) | `core/accumulators/flex_accumulator.*`, `token_layout.*` | MOVE TO SHARED SPEC + REIMPLEMENT IN RUST | Same as above, plus layout mapping group/index to token is a second spec surface |
| VarAccum / VarWidths (dense/adaptive variable widths) | `core/architectures/dense/var_accum.*` | MOVE TO SHARED SPEC | Widths logic duplicated in loader (`embWidths` vs `gw`) with no spec doc; must freeze |
| RelationalMixer forward (QKV matVec, QK^T, gabS, dynamic bias clamp +-0.25, gate clip/hard_sigmoid, residual alpha) | `core/architectures/relational/relational.cpp`, `training/models/*` | MOVE TO SHARED SPEC + REIMPLEMENT IN RUST | Gate formulae and clamp bounds are exact-mode candidates but only exist as code. Python mirror lives across several model files, not one reference kernel |
| Attention / MLP / Dense / Adaptive forwards + heads (tanh value, linear WDL, clip01 hidden) | `core/architectures/*`, `training/models/*` | MOVE TO SHARED SPEC (per-arch slices) | Same pattern: math is stable, spec is missing. Adaptive threshold comparison and refine_precision are the highest-risk parity points |
| Quantization (symmetric per-group, round-half-away via lround/np.round, clamp [-127,127]/[-32767,32767], int32 acc, dequant then clip) | `core/accumulators/*`, `training/export/*`, `training/export/quantize.py` | MOVE TO SHARED SPEC | Rounding mode is the cross-language trap: C++ `lround` vs Python `np.round` (bankers) vs Rust `round()`. Must pin one mode in numerical contract and fix all three |
| simd.h matVec/matMul/clippedRelu | `core/simd/simd.h` | REIMPLEMENT IN C++ (split ref vs opt) + REIMPLEMENT IN RUST | Today scalar-only despite the name. No AVX2 path, no dispatch, no benchmark split scalar vs SIMD. Keep scalar as reference, add opt kernel behind dispatch |
| Evaluator (refresh/evaluate, RelationalEvaluator with context) | `core/inference/evaluator.*`, relational header | KEEP + REIMPLEMENT IN RUST | C++ evaluator is correct thin glue. Rust has no evaluator at all, only batch dataset serving |
| Context compute (ContextSpec::kDim=8) | `core/architectures/relational/context.*` | MOVE TO SHARED SPEC | Tiny but load-bearing for mixer parity |
| ModelSpec struct + arch ids + version strings | `core/architectures/base/architecture.h`, `core/model_io/*` | MOVE TO SHARED SPEC | Arch ids (RUNE-SFNN/MLP/ATTN/GAB, RUNE-REL-02, RUNE-03-*, RUNE-04, RUNE-05) and `arch_version` handling scattered across loader branches; needs `spec/architecture/` canonical table |
| .rune file framing (magic RUNE, u32 LE header len, JSON header, payload order emb0..7 then arch tensors, LE floats) | `core/model_io/model_io.cpp`, `model_save.cpp`, `training/export/export.py` | MOVE TO SHARED SPEC + REIMPLEMENT IN C++ (harden) + REIMPLEMENT IN RUST | Works but incomplete as a contract: `format:1` everywhere, checksum only for dense/adaptive, no format_version/architecture_version/feature_version split, no endianness statement, minimal malformed-input validation, `isSupportedVersion` gate unclear. v0.10 must ship `spec/model-format/` with magic/version/checksum rules and make Python/C++/Rust agree on one hash |
| model_factory create* | `core/model_io/model_factory.*` | KEEP | Fine as C++ construction layer once spec pins shapes |
| Python training models + losses + trainer | `training/` | KEEP (as reference, not runtime) | Training stays Python. Only the forward reference path needs freezing for triangle tests |
| Rust data engine (parse, validate, dedup, filter, shard, format, pipeline) | `rust-data/crates/rune-data-*` | KEEP (as data plane) + DUPLICATE INTENTIONALLY (board/features reimplemented in runtime) | Data plane is good and must not be rewritten. Runtime needs its own board/features/kernel crates so data and inference do not share mutable state |
| Benchmarks (cpp_bench, stage_bench, infer_bench, quant_bench, bench.py) | `benchmarks/`, `tools/benchmark/` | REIMPLEMENT IN C++ + REIMPLEMENT IN RUST (unified rune-bench) | Each measures something useful but none is cross-language, none fixes positions, warm/cold, threads, or cycles/eval. Replace with one suite over `benchmark/positions/` |
| Bindings (rune_bindings + arch bindings) | `bindings/` | KEEP | Needed for Python vs C++ triangle; no Rust FFI in v0.10 |
| Tests (cpp suite, pytest suite, rust integration) | `tests/`, `rust-data/.../tests` | KEEP + MOVE TO SHARED SPEC (vectors) | Good coverage of single-language correctness, zero cross-language vectors. Add `spec/test-vectors/` consumed by all three |
| Tools (screening, active, analysis, dataset, match) | `tools/` | KEEP | Research tooling, out of runtime parity scope |
| Configs v09 + teacher/target pipeline docs | `configs/v09/`, docs v09 | KEEP (frozen) | v0.10 does not relabel teachers; dataset/model compat check only validates feature_version |

## 2. Canonical-source gaps (why v0.10 exists)

1. No `spec/` directory. Feature IDs, accumulator layout, token layout, gate formulae, quant rules, model framing all live as code triplets.
2. Rounding is unspecified. `lround` vs `np.round` differ on .5; Rust `round` matches `lround` (half away) but nobody pins it.
3. No golden vectors. Three implementations agree by inspection, not by shared fixture.
4. Model hash disagrees by construction: Python hashes payload for dense/adaptive only, C++ verifies only dense/adaptive, embedded scales stored differently (`scales` map vs per-group float), `format` never versioned per layer.
5. Rust has no runtime. `rune-data-core` features/board serve labeling, not evaluation; no accumulator, mixer, head, model loader, evaluator, or SIMD story.
6. C++ has no kernel split. `simd.h` is scalar reference masquerading as SIMD; no dispatch, no fallback policy, no scalar-vs-SIMD benchmark.
7. No tolerance policy. Bit-exact stages and float-tolerant stages are not distinguished anywhere.
8. No differential harness. `rune-diff`, intermediate dump, stage-by-stage compare do not exist.

## 3. What v0.10 must not rewrite

- C++ board/movegen, feature math, accumulator math, mixer math: freeze semantics, do not redesign.
- Python training loop, losses, active/screening machinery: untouched except export + reference forward.
- Rust data pipeline, `.rune-data` format, sharding, teacher lookup: profile-gated fixes only.
- Match harness, calibration/analysis tools: reuse as-is for validation views.

## 4. Carry-forward UNKNOWNs (still open after audit)

- Whether `np.round` vs `lround` has already poisoned any shipped quantized model (needs quant parity sweep).
- Whether adaptive threshold float comparison is deterministic across compilers (needs routing parity test).
- Whether dynamic-bias clamp +-0.25 hides a larger layout disagreement (needs mixer stage dump).
- Whether Rust SIMD will pay off before v1.0 (explicitly a failed-experiment candidate, not a commitment).
