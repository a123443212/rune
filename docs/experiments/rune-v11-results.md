# RUNE v0.11 results — inference/compiler release

Target giữ: RUNE-ATTN-GAB 8x32 fp32/int8. Không architecture mới.

## Definition of Done

```text
✓ v0.10 audit (docs/experiments/rune-v11-v10-audit.md)
✓ RUNE IR v1 (training/compiler/ir.py, crates/rune-ir, core/compiler/ir.*)
✓ compiler pipeline parser/verifier/optimizer/planner/backend/artifact (Python + Rust, C++ consume)
✓ model specialization theo arch/tokens/dim/dtype/quant/ISA
✓ weight packing v1 deterministic row-major-aligned32
✓ memory planning arena 7-8KB reuse + alignment 32
✓ specialized kernels small-matrix 8x32/8x8/128x256/32x128
✓ optional kernel fusion 5 nhóm, chỉ khi benchmark thắng
✓ C++ backend portable scalar + AVX2 dispatch + specialized templates
✓ Rust backend safe reference + target-specific path, unsafe isolate simd lane
✓ portable fallback (generic loader đọc compiled weights, ISA check fail rõ ràng)
✓ compiled artifact format + naming network.<isa>.rune
✓ compiler cache sha256(src|arch|quant|isa|compiler)
✓ golden vectors (v10 giữ + compiled diff 5 trials PASS)
✓ differential tests generic/compiled, stage-wise không chỉ final scalar
✓ fuzz/property 200 trials, invalid reject rõ ràng
✓ cross-language parity (shared kernel plan, bit-identical lanes giữ)
✓ CPU profiling (per-stage us, arena bytes, artifact bytes, compile <1s)
✓ end-to-end benchmark (micro 3-4x, full_eval bound bởi feature/accum)
✓ performance regression tracking (gates warn 5% latency/NPS, 20% code size, CI smoke)
✓ complete documentation (architecture, ir, compiler, specialization, benchmarks, artifact, audit, results, failed)
```

## Experiment matrix (representative, không full Cartesian)

```text
R0 generic vs R1 specialized: fused 14-19us vs generic mixer+head 58-77us (~3-4x scalar)
K0 generic small-matrix vs K1 specialized vs K2 fused: smallmat 32x32 parity 1e-6, fused qkv/score/mix thắng rõ
Q0 FP32 vs Q1 INT8: int8 compile PASS, torch diff <0.01-0.05, no drift trên fixtures
B0 C++ vs B1 Rust: không verdict ngôn ngữ, cùng plan cùng algorithm
T0 portable vs T1 AVX2: avx2 artifact build đúng, loader fail rõ trên non-avx2 CPU (fallback portable)
```

## Core benchmark (8x32, scalar, ranges)

feature 7.4us, refresh 12.4us, update 0.9us, tokenize 0.35us, QKV fused 10-17us, score+bias+gate 1.1-1.6us, mix+residual 1-3.4us, mixer generic 23-32us, head generic 34-45us, full_eval 68-90us.

## 15 câu hỏi

1. Generic overhead nằm ở đâu? Dispatch mỗi matVec, shape động mỗi call, Vec alloc 5+2 mỗi forward, q/k/v/s/g/y copies, int8 dequant lúc load, HashMap scales lookup, head+mixer 80%+ full_eval.
2. Specialized nhanh hơn bao nhiêu? Micro fused 3-4x scalar (14-19us vs 58-77us). Full_eval ít hơn vì feature/accum không đổi.
3. Fusion gain thực tế? linear+bias+clip, score+bias+gate, mix+residual, QKV đều thắng đo được. Không fuse mù.
4. Weight packing có đáng? Aligned row-major đáng (deterministic, rẻ). Transposed copy không đáng cho 32x32 (failed log).
5. Memory planning giảm bao nhiêu? Arena 7-8KB duy nhất, 0 alloc/eval (trước: 7 Vec alloc/forward). Copies giảm từ 6 passes xuống 3 fused passes.
6. C++ vs Rust specialized? Cùng plan, cùng algorithm, không verdict. Rust unsafe isolate, C++ templates fixed-size.
7. Portable vs target-specific? Portable +1.1% bytes, chạy mọi CPU. AVX2 artifact fail rõ trên scalar build, đúng thiết kế. AVX-512 chưa ship vì chưa đo được gain.
8. INT8 có giữ parity? Có trên fixtures (diff 0, torch <0.01). High-magnitude/near-zero/threshold tests pass.
9. Adaptive có compile hiệu quả? Route/cheap/refine tách kernels, thresholds constant-fold, semantics giữ. Absolute savings với trained head vẫn cần đo.
10. Code size tăng bao nhiêu? Vài KB kernels + 4.8KB plan/header (+1.1% artifact). Không explosion.
11. Compile time bao nhiêu? Python <1s, cache deterministic. Không 10-phút compile.
12. End-to-end NPS tăng bao nhiêu? Evaluator micro 3-4x, full_eval bound bởi feature/accum nên gain nhỏ hơn. Engine search NPS chưa đo ở v0.11 (cần harness riêng), đã ghi rõ.
13. Optimization nào chỉ đẹp micro? In-place toàn phần, interleaved layout, transposed packing: micro không thắng hoặc rủi ro > lợi, đã log failed.
14. Compiler nào cho v1.0? Python reference + Rust shared plan + C++ consumer. Không duplicate logic, boundary là artifact + plan JSON. Đóng băng IR v1 + kernel ids + packing v1.
15. Đóng băng gì? Feature IDs, row-major, half-away, framing, routing >= NaN→refine, per-stage eps, vector format, IR v1, kernel ids, packing v1. Thay đổi có kiểm soát: header hash full ở v1.0, scales strict equality từ v0.11.
