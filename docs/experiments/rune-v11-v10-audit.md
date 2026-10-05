# RUNE v11 — v10 audit

Ngày: 2026-10-05
Phạm vi: RUNE v0.10 (spec, model format, numerical contract, C++ runtime, Rust runtime, Python exporter, scalar kernels, SIMD kernels, differential testing, profiling, benchmark results, architecture variants, quantization)
Mục đích: chốt KEEP / REMOVE / REWORK / SPECIALIZE / GENERATE / UNKNOWN cho v0.11 compiler release. v0.11 giữ architecture tốt nhất của v0.10, không thêm attention/SSM/MoE/representation/loss/search/policy mới.

## 1. Kết luận nhanh

Architecture giữ lại cho v0.11: RUNE-ATTN-GAB 8x32, head value_wdl 128->32, gate clip, fp32 + int8. Đây là model được đo nhiều nhất ở v0.10 (small-gab-fp32, 2000 iters, per-stage us, scalar vs SIMD hai ngôn ngữ, triangle 9/9 PASS). Các biến thể REL-02, dense, adaptive giữ ở dạng portable, không phải specialization target chính.

Generic overhead lớn nhất nằm ở: dispatch mỗi matVec, shape kiểm tra động mỗi forward, packing/quant mỗi eval, temporary Vec allocation trong mixer/head, Q/K/V lặp 3 lần qua cùng một pattern, score+bias+gate+mix chạy rời rạc, head 128x256 + 32x128 chạy generic matVec.

## 2. Bảng phân loại

| Hạng mục | Phán quyết | Lý do |
|---|---|---|
| spec/ Layer 1 semantics, VERSIONS, numerical-contract per-op eps | KEEP | Đã là source of truth, 3 ngôn ngữ defer về nó. v0.11 chỉ thêm IR version độc lập, không sửa semantics hiện có |
| model format v2 framing, magic, header_len, payload order, fnv1a64 payload | REWORK | Giữ framing để tương thích, nhưng phải thêm source hash bao header, strict scales equality, compiled artifact fields. Hiện hash chỉ cover payload nên đổi scale không đổi hash |
| Python exporter export.py symmetric_scale, quantize_array, header dual names | REWORK | Giữ math, sửa strict equality scales vs quantization_metadata, thêm canonical graph export bên cạnh .rune |
| C++ ref_kernels scalar exact | KEEP | Làm oracle cho compiled kernels, giữ nguyên để differential test |
| C++ simd_kernels AVX2 matVec dispatcher | SPECIALIZE | Chỉ specialize matVec đơn lẻ, chưa cover QKV fused, score+bias+gate, head fused, small-matrix 8x32. Giữ dispatcher cho portable, thêm specialized path cố định shape |
| Rust rune-kernel scalar + simd.rs AVX2+FMA lane | SPECIALIZE | Bit-identical với C++ là tài sản lớn, giữ. Cần thêm fused/smallmat/quant specialized, giữ safe wrapper isolate unsafe |
| C++ Evaluator + GroupedAccumulator refresh/applyDiff/tokens | SPECIALIZE | Đúng semantics, nhưng tokenBuf_[8*32] cố định 256 trong khi Rust dùng tokens*dim động. Specialize theo tokens/dim cố định, precompute offsets, batch sparse update |
| Rust Evaluator + mixer.rs forward alloc Vec mỗi lần | SPECIALIZE | q/k/vv/s/g/y alloc mỗi forward là overhead lớn nhất. Chuyển sang arena scratch reuse, fused kernels |
| Head forward flatten + h1 + h2 + tanh + wdl | SPECIALIZE | Head là stage SIMD-friendly nhất (~7x đã đo). Tách linear+bias+clip thành fused kernel, precompute packed W |
| Accumulator int8/int16 paths | SPECIALIZE | Quant paths tồn tại nhưng Rust evaluator reject dense-b-int8, adaptive. Specialize input scales, accum width, requant, clamp ngoài innermost loop |
| Feature extraction GroupedFeatureSet | KEEP | startpos 93 features, refresh==incremental đã PASS. Chỉ packing index representation cho hot path |
| Architecture table RUNE-SFNN/MLP/ATTN/ATTN-GAB/REL-02/03/04/05 | KEEP cho ATTN-GAB, REMOVE khỏi specialization target cho 03/04/05 | 03/04/05 giữ portable fallback, không generate specialized kernels ở v0.11 trừ adaptive cheap/refine/route tách riêng khi có benchmark chứng minh |
| Adaptive routing threshold >=, NaN->refine | KEEP | Exact, đã sweep 9 thresholds 100% agreement. Compiler chỉ constant-fold threshold, không đổi semantics |
| Golden vectors spec/test-vectors/v10 + tools/golden | REWORK | Giữ oracle, thêm compiled vectors (generic vs compiled, C++ vs Rust compiled) cùng tolerance policy |
| cross_check.py triangle + rune_diff stage self-check | REWORK | Mở rộng thành 4 chiều: generic/compiled x C++/Rust, compare từng stage không chỉ final scalar |
| run_bench.py + rune_bench_v10 + rune_stage_bench | REWORK | Giữ method same-model same-hardware, thêm modes reference/generic/specialized, metrics cycles/ns/NPS/bytes/code-size/scratch |
| Profiling docs v04/v05/v06 + v10-benchmarks ranges | KEEP | Số liệu scalar 63-75us, SIMD 15-20us, head 30-36us->3.6-5.9us là baseline. v0.11 phải đo lại cùng máy, report range không digit đơn |
| Quantization symmetric half-away, bound 127/32767 | KEEP | Đã pin np.round ban, lround/round/floor-ceil thống nhất. INT8 specialization giữ parity, test drift riêng |
| Dual scales maps (scales vs quantization_metadata.scales) | REMOVE | Latent divergence trap, hai reader resolve khác nhau khi disagree. v0.11 loader strict equality, v1.0 hash header |
| Format 1 legacy aliases arch/arch_version/feature_set/tensors/scales/checksum | REMOVE khỏi writer | Chỉ accept on read, writer chỉ emit canonical v2 + compiled fields |
| C++ model_io RuneFile union isInt8/isInt16/isFlex/isDense/isAdaptive | REWORK | Giant variant object, load mọi path. Compiler tách per-arch loader, fail-closed unknown op |
| Rust model loader HashMap arrays + Vec<f32> dequant lúc load | SPECIALIZE | Dequant int8->fp32 lúc load rồi chạy fp32 kernels làm mất lợi ích INT8. Specialize giữ int8 packed + scales riêng |
| Thread scaling benchmark, cache warm vs cold | KEEP | Sub-linear, 10-15% warm/cold diff đã ghi nhận. v0.11 giữ, không claim single-cache |
| Data engine rust-data + rune-data workspace | KEEP | Không bottleneck ở v0.10 scale, không rewrite. Chỉ tách runtime helpers đã làm |
| No FFI C++<->Rust, boundary là .rune bytes | KEEP | v0.11 boundary là compiled artifact + shared kernel plan JSON, vẫn không FFI |
| Pybind bindings | KEEP | Không chạm ở v0.11 trừ khi cần expose compile/inspect |
| Unknown arch/dtype/ISA handling fail-closed | KEEP | Mở rộng cho compiler: unsupported shape/dtype/ISA/malformed/checksum/unknown-op reject rõ ràng, không silent fallback semantics khác |

## 3. Generic overhead đã xác định

```text
function dispatch: kern::matVec -> activePath() check mỗi hàng, Rust active_path() atomic load mỗi forward
dynamic shape handling: rows/cols/tokens/dim truyền int mỗi call, loop bounds không const, vector len check mỗi kernel
unnecessary memory movement: flatten tokens row-major copy, q/k/v 3x256 floats ghi rồi đọc lại, s 8x8 ghi đọc, g 8x8 ghi đọc, y 256 ghi đọc, h1/h2 Vec alloc
repeated packing/unpacking: int8 dequant lúc load rồi chạy fp32, tokensClip mỗi eval, scales lookup HashMap mỗi tensor
temporary buffers: mixer forward 5 Vec alloc, head forward 2 Vec alloc, C++ arch forward alloc trong relational/dense/adaptive
```

Đo v0.10 cho thấy head + mixer chiếm 80%+ full_eval_refresh_us. Feature extract 7-10us, refresh 18-20us, update 0.8-1.3us, tokenization 0.3-0.6us, mixer 35-47us scalar, head 60-74us scalar. SIMD giảm mixer+head xuống 5-12us + 3.6-5.9us. Phần còn lại sau SIMD chính là overhead cần specialize: accumulator indexing, feature diff, dispatch, copies.

## 4. Danh sách SPECIALIZE cụ thể cho v0.11

```text
accumulator update: group/offsets/increment pattern compile-time, batch cùng group
tokenize clip: fused dequant+clip, static bounds
Q/K/V: một fused QKV kernel 8x32 thay vì 24 matVec calls rời
score: QKT 8x8 + gab add + gate trong một microkernel, không generic GEMM
mix: gate x V 8x32 fused + residual x+alpha*y
head: W1+bias+clip fused 128x256, W2+bias+clip fused 32x128, wvo dot + tanh, wwdl 3x32
quant: input/weight scales, accum width i32, clamp, requant hoisted khỏi loop
memory: arena scratch Q/K/V/scores/gate/tmp, reuse, alignment 32B, lifetime plan
layout: row-major giữ, thử packed/transposed cho Wq/Wk/Wv/W1/W2, interleaved chỉ khi kernel yêu cầu
adaptive: cheap/refine/route tách kernel riêng, không dynamic graph traversal
```

## 5. Danh sách GENERATE

```text
RUNE IR v1 JSON: ops FeatureUpdate->AccumulatorUpdate->Tokenize->Q->K->V->Score->Bias->Gate->Mix->Residual->Head
canonical graph manifest từ Python exporter
kernel plan JSON shared C++/Rust: kernel id, shape, dtype, packing, ISA, fusion plan
generated C++/Rust small-matrix kernels cho 8x32, 32x32, 32x128, 128x32, 8x8
compiled artifact network.<isa>.rune + manifest
memory plan + build cache keyed model hash + arch + precision + ISA + compiler version
```

## 6. UNKNOWN cần experiment trả lời

```text
fusion score+bias+gate có nhanh hơn rời rạc trên AVX2 thực tế hay chỉ giảm op count
weight packing transposed có lợi cho 32x32 nhỏ hay chỉ tốn L1
INT8 head specialization có giữ parity trong bound hay drift ở high-magnitude eval
Rust SIMD có gain thêm khi fused hay đã bão hòa ở 15-18us
AVX-512 có gain thực tế trên 8x32 hay chỉ tăng code size + icache pressure
arena reuse giảm bao nhiêu alloc/copy ở engine level, có chuyển thành NPS không
```

## 7. RQ mapping

RQ1 overhead từ abstraction: dispatch + dynamic shape + Vec alloc + copies, đo ở section 3.
RQ2 specialized vs generic: baseline SIMD 3.7-4.4x đã chứng minh generic->SIMD, v0.11 cần specialized small-matrix vs generic GEMM.
RQ3 fusion: candidate linear+bias+clamp, score+bias+gate, mix+residual, chỉ fuse khi benchmark thắng.
RQ4 codegen giữ parity: yêu cầu reference graph -> compiled graph per-stage parity, golden vectors 5 chiều.
RQ5 shared plan: kernel id/shape/dtype/packing/ISA/fusion dùng chung, không drift.
RQ6 hardware specialization: AVX2 vs portable vs AVX-512 đo cùng máy cùng model, report range.

## 8. Quyết định đóng băng

Giữ nguyên: feature IDs, tensor layout row-major, quant half-away, framing magic+header_len, routing >= NaN->refine, tolerance policy per-stage, vector format.
Thay đổi có kiểm soát: header hash cover header+payload ở v1.0, hiện thêm source_hash + strict scales check ở v0.11 compiled artifact. Format 1 read-only.
Không thêm architecture mới ở v0.11.
