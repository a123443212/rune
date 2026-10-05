# RUNE v0.10 benchmarks

Method: `tools/bench/run_bench.py` drives `build/rune_bench_v10`
(C++) and `target/release/rune bench` (Rust) on the same model
(`small-gab-fp32`) and same machine. Positions are fixed in
`benchmark/positions/` (quiet/tactical/endgame/king_attack/
random). Report per-stage us + eval/sec, never a single score.

## Per-stage (C++, scalar path, O2)

```text
feature_extract_us       ~7-10
accumulator_refresh_us   ~18-20
accumulator_update_us    ~0.8-1.3
tokenization_us          ~0.3-0.6
mixer_us                 ~35-47
head_us                  ~60-74
full_eval_refresh_us     ~95-112
full_eval_incremental_us ~77-87
```

## End-to-end

```text
C++  refresh    ~8945-10514 eval/sec
C++  incremental ~11524-13824 eval/sec
Rust release     ~9023 eval/sec  (2000 iters, startpos refresh)
Rust/C++          0.99x  (scalar vs scalar, release)
Rust debug       ~870 eval/sec  (12x slower — never used for claims)
```

## S1: scalar vs SIMD matrix (same machine, same day, small-gab-fp32)

Method: `run_bench.py --cpp-bench-avx2` (AVX2 build in a scratch
dir; default `build/` stays scalar) plus `rune bench --kernel
scalar|simd`. Full-refresh eval, 2000 iters, ranges over repeats
(machine noise is ~±20% at these timescales, so ranges, not
digits):

```text
               scalar      SIMD (AVX2+FMA)   speedup
C++            ~63-75 us  ~19-20 us         ~3.7x
Rust           ~64-71 us  ~15-18 us         ~4.4x
```

Stage split under SIMD (C++ AVX2 build, us): mixer 5-12
(was 14-24), head 3.6-5.9 (was 30-36). The head (128x256 +
32x128) is the most SIMD-friendly stage, ~7x. Feature
extraction and accumulator stay scalar in both languages.

Cross-language kernel equivalence (same LCG streams, 8 shapes
1x1..128x256, bias and no-bias):

```text
regime              C++ worst   Rust worst  bound
in-network (|m|<=.08, |v|<=1)  1.67e-06    1.67e-06    1e-5
stress (|m|,|v|<=1)            3.81e-05    3.81e-05    1e-4
mixer.json golden weights      bit-exact vs scalar at 1e-6
```

Per-shape worsts match digit-for-digit: the two AVX2 lanes are
bit-identical implementations of one algorithm (8-wide FMA,
scalar tail, left-to-right horizontal sum). SIMD-vs-scalar
difference is FMA rounding plus 8-way reassociation — a property
of the algorithm, equal in both languages. End-to-end SIMD
triangle (`cross_check.py --cpp-path avx2 --rust-kernel simd`):
9/9 PASS, max C++↔Rust diff 5e-07, values equal to scalar to
6 decimals.

What this does NOT claim: neither language is "faster". SIMD
Rust measured 15.3 vs SIMD C++ 20.1 in one run and the reverse
gap in another — that is noise on 30-40ms totals, not a verdict.
The finding is parity: same algorithm + same ISA ⇒ same speed,
and the speedup (3-4x) belongs to the kernel, not the language.

## Threads

`rune_bench_v10 --threads 1/2/4/8` scales sub-linearly (memory
bound accumulator + head). Numbers are machine-specific; the
suite prints MT ms + NPS per run instead of asserting scaling.

## Cache

Warm (repeated startpos) vs cold-ish (walk over position files)
differ by ~10-15%. Both are reported; single-cache conclusions
are banned.

## What this does not claim

- No language-speed verdict: scalar is a tie (~0.9-1.0x both
  ways across runs), SIMD is a tie (~15-20us both). Noise
  dominates any gap at these timescales.
- No engine NPS verdict: Rust has no search; engine NPS is C++
  only, evaluator throughput is the shared metric.
- No cross-machine comparison: all ratios are same-hardware.

## Adaptive cascade cost model (C++ kernels, adaptive-fp32)

`tools/bench/adaptive_sweep.py` over 21 benchmark positions:
difficulty parity Python↔C++ 3.36e-08, refined-value parity
1.86e-08, routing agreement 100% at 9 swept thresholds with
refine rates 1.00 → 0.95 → 0.76 → 0.62 → 0.19 → 0.00.
Kernel costs (`rune_stage_bench`): cheap forward 7.7us, refine
forward 45.1us, route 0.002us. Expected cost = cheap + rate ×
refine: at rate 0.62 ≈ 35.7us vs always-refine ≈ 52.7us
(~32% saved). Torch-side timing is framework-overhead-dominated
(~1ms) and is reported by the tool but never used for claims.
Caveat: the fixture is untrained so difficulties cluster in
[0.058, 0.079]; absolute savings with a trained difficulty head
remain to be measured — the machinery and the agreement are
proven, the rate is not.
