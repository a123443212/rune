# Compiled artifact format (v0.11)

Framing giữ nguyên `.rune` để tương thích: magic `RUNE`, u32 LE header_len (<=1000000), JSON header, payload. Compiled artifact là `.rune` hợp lệ + extra header fields. Generic loader vẫn đọc được weights (portable fallback). Specialized loader kiểm tra thêm plan + ISA + hashes.

## Required compiled fields

```text
compiled            bool true
compiled_kind       "compiled-v11"
rune_ir_version     "1.0"
compiler_version    "0.11.0"
target_isa          portable|avx2|avx512
target_cpu          generic-x86-64 (không hard-code model name nếu không cần)
kernel_plan         [{op, kind, kernel_id, shape, dtype, packing, isa, fusion_group}]
fusion_plan         [{id, ops, kernel}]
memory_plan         {arena_bytes, alignment, buffers, strategy, in_place}
packing_meta        {packing_version: 1, isa, layouts}
precomputed         {quantization, quant_scales, zero_points, accum_width, clamp, static_gab_*, routing_*, alpha, bias_clamp}
source_hash         sha256(header subset + payload)[:16]
kernel_plan_hash    sha256(kernel_plan + memory + target)[:16]
spec_version        "RUNE-10"
compile_time_utc    int
model_hash          fnv1a64 payload (giữ để tương thích v0.10)
```

## Naming

```text
network.rune             portable compiled (hoặc generic gốc)
network.avx2.rune        specialized AVX2
network.avx512.rune      specialized AVX-512 (khi có)
```

Không overwrite portable artifact.

## Loader checks

- magic, header_len bound, payload length = sum tensor bytes, dims trong phạm vi, dtype known.
- compiled == true, ir version == 1.0, isa supported, CPU hỗ trợ (avx2 artifact trên non-avx2 fail rõ ràng).
- kernel plan non-empty, source hash + model hash verify, checksum mismatch fail closed.
- unknown op/shape/dtype/ISA/malformed/checksum → reject, không silent fallback semantics khác.
- portable fallback: nếu specialization không khả dụng, chạy generic path trên cùng weights, behavior rõ ràng, không chạy ISA không được hỗ trợ.

## Reproducibility

Cùng model hash + compiler version + target ISA → cùng bytes (test `test_artifact_reproducible`). Metadata reproduce: compiler commit, model hash, spec hash, target, optimization settings. Cache key = sha256(src|arch|quant|isa|compiler).

## Known limitation kế thừa

v0.10 hash chỉ cover payload. v0.11 thêm source_hash cover header subset + payload, nhưng full header hash vẫn để v1.0 (hash header+payload hoặc second header checksum). Scales dual-maps strict equality ở loader mới.
