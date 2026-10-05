# RUNE v0.10 results (running log)

## Correctness (C0–C3)

- C0 C++ reference: `build/rune_tests` ALL PASS including new
  `runV10Tests` (quant half-away, startpos 93 features,
  refresh==incremental, NaN routing, 7 fixture loads, make/unmake).
- C1 Rust reference: `cargo test --workspace` — kernel 6/6,
  simd 5/5, model 4/4, runtime 8/8 PASS (golden features,
  refresh==incremental, make/unmake, fixtures, triangle value,
  quant closeness, SIMD-vs-scalar tol, unsupported-arch
  fail-closed).
- C2 C++ optimized: AVX2 matVec path exists behind dispatch;
  scalar-vs-dispatched diff is 0 on golden positions
  (`rune_diff --mode tolerant` PASS).
- C3 Rust optimized: DONE — AVX2+FMA lane in
  `crates/rune-kernel/src/simd.rs`, bit-identical to the C++
  lane on all fuzz shapes, ~4.4x eval speedup measured.

## Parity (P0–P2)

- P0 Python vs C++: triangle on `small-gab-fp32` startpos —
  Python -0.0197625, C++ -0.019763, diff ~7e-7.
- P1 Python vs Rust: same position, Rust -0.019763, diff ~7e-7.
- P2 C++ vs Rust: `cross_check.py` over 9 quiet/tactical
  positions, all PASS, max C++↔Rust diff 4.9e-07 (tol 2e-5).

## Performance (S0–S1)

Release numbers, same machine, `small-gab-fp32`, 2000 iters:

```text
C++  full_eval_refresh_us     ~111.8
C++  full_eval_incremental_us  ~86.8
Rust us_per_eval               ~110.8  (release)
Rust/C++ ratio                  0.99
```

Debug Rust is ~12x slower; release closes the gap. SIMD (S1)
is now measured both languages (details in
`docs/performance/rune-v10-benchmarks.md`): C++ 63-75us →
19-20us, Rust 64-71us → 15-18us; lanes bit-identical, end-to-end
SIMD triangle 9/9 PASS. Same algorithm + same ISA ⇒ same speed;
no language verdict in either direction.

## Model formats (M0–M2)

- M0 FP32: tiny-mlp, small-gab, rel-08x32, dense-b, adaptive
  all load in Python+C+++Rust, same hash.
- M1 INT8/INT16: small-gab-int8/16 load everywhere; dense-b-int8/16
  shipped and proven Python↔C++ (`dense_check.py` 9/9, gap ~7e-06;
  C++ `testDenseQuantCloseness`; pytest `test_v10_debts.py`);
  quant vectors pin half-away. Rust loads dense fixtures but its
  evaluator rejects them by allowlist (correct: unimplemented,
  not silently wrong).
- M2 adaptive: adaptive-fp32 loads; routing NaN→refine pinned;
  full cascade sweep DONE (`adaptive_sweep.py`): difficulty parity
  3.36e-08, value parity 1.86e-08, 100% routing agreement at 9
  thresholds, rates 1.00→0.00, cost model cheap 7.7us + refine
  45.1us. Absolute savings with a trained difficulty head remain
  to be measured (fixture difficulties cluster in [0.058, 0.079]).

## Data engine

No rewrite. `rune-data` workspace still builds; new
`crates/rune-data` adds only the runtime-separated helpers.
No profiling-driven bottleneck was found at v0.10 scale, so
no streaming/mmap change was made. Debt repaid: the 7 failing
`rune-data-cli` integration tests (missing `two_games.pgn`
fixture, never committed) now pass 9/9 with a committed 2-game
fixture; CI restored to `cargo test --workspace` for the legacy
workspace so this class of slip cannot recur silently.
