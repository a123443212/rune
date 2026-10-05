# RUNE v0.11 benchmarks

Method: cùng model (small-gab-fp32), cùng máy, cùng threads/precision. Ranges over repeats, noise ±20% ở timescale này. Report per-stage us + eval/sec + bytes, không single score.

## Per-stage generic (C++ scalar, O2, build/rune_bench_v10)

```text
feature_extract_us       ~7.4
accumulator_refresh_us   ~12.4
accumulator_update_us    ~0.9
tokenization_us          ~0.35
mixer_us                 ~32
head_us                  ~45
full_eval_refresh_us     ~87
full_eval_incremental_us ~90
eval_per_sec             ~11100-11500
```

## Fused specialized (C++ scalar, build/rune_bench_compile, 2000 iters)

```text
qkv_fused_us             ~10-17
score_bias_gate_us       ~1.1-1.6
mix_residual_us          ~1.0-3.4
fused total              ~14-19
mixer_generic_us         ~23-32
head_generic_us          ~34-45
speedup fused vs generic mixer+head ~3-4x
arena_bytes              7328
```

Fused thắng vì: đọc tokens một lần, loop bounds const, không Vec alloc, scores/gate/mix giữ trong L1, head linear+bias+clip một pass.

## End-to-end compiled vs generic

Compiled evaluator (Rust fused 8x32 path + generic fallback) numerical parity 0.00e+00 trên random tokens (cùng weights). Latency end-to-end phụ thuộc accumulator + feature (không đổi) nên gain nhỏ hơn microbenchmark: micro 4x không chuyển thành 4x full_eval vì feature/refresh/tokenize (~20us) không specialize sâu ở v0.11. Engine-level NPS/search validation: chưa đo search ở v0.11 vì cần engine harness riêng; evaluator throughput là metric shared.

## Artifact size

```text
small-gab-fp32.rune generic      434625 bytes
network.portable.rune compiled   439476 bytes (+1.1%)
```

Code size: smallmat+fused+quant+accum vài KB, không explosion. Compile time Python <1s. Cache deterministic.

## Threads / cache

Giữ kết luận v0.10: MT sub-linear (memory bound), warm vs cold 10-15%. Không claim single-cache. Performance CI chỉ smoke/regression signal vì hardware noise. Gates: latency/NPS warn 5%, code size warn 20%, không fail CI vì vài phần trăm noise.

## INT8

small-gab-int8 compile portable PASS, diff 0.00e+00 (cùng weights path). Torch fp32 vs int8 value diff <0.01 (dense-b), <0.05 spot check. High-magnitude/near-zero/threshold-boundary tests: threshold >= exact, NaN→refine pinned, clip exact.

## Cross-compiler

C++ generic vs Rust generic tie ở v0.10 (0.99x). v0.11 không kết luận Rust vs C++: so sánh kernel design (fused vs generic), không phải ngôn ngữ. Shared plan đảm bảo cùng algorithm.

## Profiling

L1/L2/LLC/bandwidth: chưa có profiler tích hợp ở v0.11, chỉ arena bytes + qualitative icache (kernels nhỏ). Không claim cache-miss numbers khi không đo được trên mọi máy.
