# RUNE v0.11 Architecture: compiled + specialized inference

Câu hỏi v0.11: generic runtime overhead nằm ở đâu, và bao nhiêu trong đó có thể biên dịch bỏ đi mà vẫn giữ semantics.

v0.10 đã chốt: spec là source of truth, C++ là production runtime, Rust là secondary + research, Python là training + reference. v0.11 giữ nguyên ba vai trò này, thêm một layer mới: compiler/specializer biến model specification thành inference artifact gắn với shape/precision/ISA cụ thể.

```text
Generic Runtime
      ↓
Model Specification (.rune fp32/int8, RUNE-ATTN-GAB 8x32)
      ↓
RUNE IR v1 (ops, shapes, dtypes, lifetimes, fusion)
      ↓
Kernel plan + memory plan (shared C++/Rust)
      ↓
Compiled artifact (network.rune / network.avx2.rune)
      ↓
C++ / Rust specialized runtime (arena, fused, static shapes)
```

## Giữ gì từ v0.10

RUNE-ATTN-GAB 8x32 head 128->32 gate clip là specialization target chính vì được đo nhiều nhất (triangle 9/9 PASS, SIMD 3.7-4.4x). REL-02 8x32 dùng chung kernels. MLP/SFNN dùng head path. Dense/adaptive/uncertainty giữ portable fallback, chỉ tách cheap/refine/route kernels khi benchmark chứng minh.

## Thêm gì ở v0.11

- `training/compiler/`: IR builder, canonical graph, packing, memory planner, kernel selector + cost model, precomputed constants, artifact writer + cache. Đây là reference compiler.
- `crates/rune-ir`: IR types + verifier + shape_key + kernel_for, dùng chung hai ngôn ngữ.
- `crates/rune-compiler`: parser từ RuneModel, verifier, optimizer (fusion), planner (kernels + arena), artifact writer (hash, cache). Không duplicate logic với Python: cùng schema, cùng kernel ids.
- `crates/rune-kernel/{specialized,fused,quant,arena}`: small-matrix 8x32/128x256/32x128, fused qkv/score-bias-gate/mix-residual/linear-bias-clip, quant dequant-clip batched, arena reuse. Unsafe chỉ trong simd lane đã có, có safe wrapper + bounds check.
- `crates/rune-runtime/{compiled,compiled_loader,accum_special}`: CompiledEvaluator dùng fused path khi 8x32, generic fallback các shape khác, grouped sparse update theo group, arena scratch.
- `core/compiler/`, `core/kernels/{smallmat,fused,quant_specialized,accum_specialized}`, `core/runtime/{arena,compiled_loader,compiled_evaluator}`: đối xứng C++, templates fixed-size, không template-metaprogramming quá mức.
- Tools: `rune-compile` (Python), `rune compile/bench-compile/inspect-compiled/diff-compiled` (Rust), `rune_compile_support/rune_bench_compile/rune_model_info` (C++).
- Tests: IR verify, packing deterministic, artifact reproducible, kernels parity + randomized + threshold boundary, golden 5 chiều, fuzz compiler, C++ test_v11.

## Không làm

Không thêm attention/SSM/MoE/representation/loss/search/policy mới. Không xây LLVM tổng quát. Không hard-code CPU model name. Không benchmark autotune mỗi startup. Không biến Rust thành toàn unsafe.

## Thành công khi nào

Ít nhất một trong hai: latency/NPS cải thiện thực tế trên CPU, hoặc giảm rõ runtime overhead/memory movement, mà vẫn giữ numerical parity per-stage.
