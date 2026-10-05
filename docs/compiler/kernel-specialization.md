# Kernel specialization

## Portable vs specialized

Portable artifact: ISA portable, chạy mọi x86-64, runtime dispatch avx2→scalar nếu build có AVX2. Specialized artifact: target_isa cố định (avx2/avx512), loader kiểm tra CPU, fail rõ ràng nếu không hỗ trợ, bỏ dynamic dispatch trên fast path. Không buộc production chỉ một target: có thể ship network.rune + network.avx2.rune song song.

## Specialization parameters

architecture, token count, token dimension, hidden dimension, dtype, quantization, ISA. Ví dụ tokens=8 dim=32 dtype=fp32 ISA=AVX2. Sau compile không kiểm tra shape động cho dims đã cố định: templates C++ `matVecFixed<32,32>`, Rust `matvec_32` với debug_assert thay vì runtime branch.

## Small-matrix kernels

Shapes thực tế duy nhất được benchmark: 8x32 (tokens), 32x32 (QKV), 8x8 (scores), 128x256 + 32x128 (head), 3x32 (wdl). Không benchmark shape không xuất hiện trong model.

Đo scalar C++ cùng máy (ranges, noise ±20%):

```text
qkv_fused_8x32        ~10-17 us
score_bias_gate_8x8   ~1.1-1.6 us
mix_residual_8x32     ~1.0-3.4 us
fused total           ~14-19 us
mixer generic         ~23-32 us
head generic          ~34-45 us
generic mixer+head     ~58-77 us
speedup fused         ~3-4x (scalar vs scalar, cùng máy)
full_eval             ~68-90 us (bao gồm feature 7us + refresh 12us + overhead)
```

Speedup thuộc về kernel design (đọc x một lần, không alloc, loop bounds const), không phải ngôn ngữ.

## Fusion

- linear+bias+clip (head): thắng rõ, head là stage SIMD-friendly nhất.
- score+bias+gate 8x8: thắng, giảm 3 passes trên 64 floats xuống 1.
- mix+residual: thắng, giữ tmp trong cache.
- QKV fused: thắng, đọc token một lần cho 3 projections.

Chỉ fuse khi benchmark thắng. Không fuse để code khó đọc.

## Weight packing

Packing v1 deterministic, versioned: row-major-aligned32 (pad cols lên bội 8), layouts ghi trong header. Transposed copy chỉ khi kernel yêu cầu. Không pack runtime mỗi eval.

## Quantized specialization

INT8 giữ int8 payload + scales riêng, dequant-clip batched theo group (8 scales), accum int32, clamp ngoài loop. Parity: small-gab-int8 vs fp32 value diff <0.05 trên fixtures (torch reference). Không để generic quant functions trong innermost loop.

## Accumulator specialization

Grouped update: gom added/removed theo group, bỏ qua group trống, offsets precomputed, batch cùng group thành một accumulation sequence. Feature IDs pack thành u8 group + u16 index arrays cho hot path. Không giant runtime object graph.

## Adaptive

Cheap/refine/route tách kernels riêng, route là float32 >= với NaN→refine, thresholds là bytes trong model. Không dynamic graph traversal. Fixture hiện tại difficulties cluster [0.058,0.079] nên absolute savings với trained head vẫn cần đo thêm.

## Runtime dispatch

Portable binary dispatch avx2→scalar qua `activePath()` (C++) / `active_path()` (Rust atomic). Specialized artifact bỏ dispatch khi shape+ISA cố định. Benchmark cả hai modes.

## Unsafe policy

Rust unsafe chỉ trong AVX2 lane đã có, isolate trong `simd.rs`, safe wrapper kiểm tra lens + has_avx2_fma fallback scalar, boundary validation, tests, benchmark justification. Không unsafe trong fused/specialized safe paths.

## Memory

Arena duy nhất ~7-8KB cho 8x32/128/32 (đo 7328-7936 bytes), alignment 32, reuse q/h1 k/h2 scores/gate, không alloc mỗi eval. Layouts thử: row-major giữ, packed aligned, col-major/transposed chỉ khi kernel yêu cầu. Không brute-force hàng trăm layouts.

## Footprint

Artifact +1.1%. Code size tăng giới hạn ở smallmat+fused+quant+accum (vài KB). Benchmark code size + icache qualitative: kernels nhỏ, không unroll quá mức. AVX-512 chưa implement vì chưa có practical benefit đo được trên 8x32 ở máy test (không có AVX-512 CPU để claim).
