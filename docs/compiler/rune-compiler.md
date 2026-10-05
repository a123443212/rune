# RUNE compiler pipeline

```text
Compiler
├── parser (model file → IR)
├── verifier (schema + shapes + isa + quant)
├── optimizer (fusion groups)
├── planner (kernel selection + memory plan + cost model)
├── backend (C++ / Rust shared plan)
└── artifact writer (packed weights + manifest + hashes + cache)
```

Không xây LLVM tổng quát. Graph nhỏ, fixed, CPU inference: rule-based selection là đủ.

## Parser

Python: `load_exported_arrays` đọc `.rune`, `build_ir(spec, metas, target)` dựng ops/tensors/target. Rust: `build_from_model(&RuneModel, isa, cpu)` cùng logic, cùng reject tokens/dim/isa/quant ngoài phạm vi. C++ không parse IR đầy đủ, chỉ đọc compiled header fields cần thiết.

## Optimizer

Fusion cố định 5 nhóm, chỉ giữ khi ops tồn tại: f_qkv (Q,K,V), f_score_bias_gate, f_mix_residual, f_head_h1, f_head_h2. Không fuse mọi thứ: chỉ fuse khi benchmark thắng, đo lại mỗi lần đổi kernel.

## Planner

`select_kernels`: shape_key từ tokens/dim/h1/h2, kernel_for ánh xạ shape→kernel id, packing row-major-aligned32, isa từ target. Tokenize chuyển sang dequant_clip khi int8/int16. `plan_memory`: arena duy nhất, alignment 32, reuse q/h1, k/h2, scores/gate, in-place gate_in_scores. `cost_model`: flops = rows*cols, fused giảm memory 0.6x, isa factor portable 1.0 / avx2 0.35 / avx512 0.28. Micro-autotune là optional: chỉ khi nhiều kernels sát nhau mới benchmark chọn trên build machine, artifact ghi rõ lựa chọn, không autotune mỗi startup.

## Backend

C++ và Rust dùng cùng kernel plan JSON: kernel id, shape, dtype, packing, ISA, fusion plan. Không drift: ids định nghĩa một lần trong `rune-ir`, cả ba ngôn ngữ map cùng tên.

## Artifact writer

Python `write_compiled`: payload packed + header JSON (compiled, kind, ir version, compiler 0.11.0, target, kernel_plan, fusion_plan, memory_plan, packing_meta, precomputed, source_hash, plan_hash, compile_time) + fnv1a64 model_hash. Rust `write_compiled` cùng framing. Cache key = sha256(model hash|arch|precision|isa|compiler). Cache deterministic, lưu JSON manifest.

## Precomputed

scales, zero_points (=0 symmetric), accum width (fp32/int32), clamp bounds, static GAB shape/max, routing thresholds, alpha, bias clamp. Không recompute trong hot path.

## Compile cost

Compile Python <1s cho small-gab (đo thực tế). Artifact +1.1% bytes (434625 → 439476) cho plan + hashes. Không compile 10 phút để gain vài ns: report luôn compile cost + artifact size + runtime gain.
