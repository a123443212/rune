# RUNE IR v1

IR version độc lập với model format version, architecture version, dataset version. Hiện tại: IR 1.0, spec RUNE-10, format 2.

## Canonical op order

```text
FeatureUpdate
→ AccumulatorUpdate
→ Tokenize
→ Q → K → V
→ Score → Bias → Gate
→ Mix → Residual
→ HeadH1 → HeadH2 → Value → WDL
```

Adaptive models thêm `Route` (cheap luôn chạy, refine có điều kiện). Không dùng ONNX: graph nhỏ, fixed, cần tensor shapes + dtype + quantization + constants + deps + lifetimes + fusion trong một JSON nhẹ.

## Schema

```json
{
  "ir_version": "1.0",
  "spec_version": "RUNE-10",
  "model": {"architecture": "RUNE-ATTN-GAB", "tokens": 8, "token_dim": 32, "dtype": "fp32", "quantization": "fp32", "gate": "clip", "alpha": 1.0, "head_h1": 128, "head_h2": 32, "threshold": 0.5, "t_high": 0.5, "t_low": null},
  "tensors": [{"name": "wq", "shape": [32, 32], "dtype": "float32", "layout": "row-major", "constant": true, "lifetime": [0, 12]}],
  "ops": [{"id": "op03", "kind": "Q", "inputs": ["tokens", "wq", "bq"], "outputs": ["Q"], "attrs": {"tokens": 8, "dim": 32}}],
  "memory": {"arena_bytes": 7936, "alignment": 32, "buffers": {}, "strategy": "...", "in_place": []},
  "fusion": [{"id": "f_qkv", "ops": ["Q", "K", "V"], "kernel": "qkv_fused_8x32"}],
  "kernel_plan": [{"op": "op03", "kind": "Q", "kernel_id": "qkv_fused_8x32", "shape": "8x32", "dtype": "fp32", "packing": "row-major-aligned32", "isa": "portable", "fusion_group": "f_qkv"}],
  "target": {"cpu": "generic-x86-64", "isa": "portable", "vector_width": 1, "dtype": "fp32", "quantization": "fp32"},
  "hashes": {}
}
```

## Verifier

Reject rõ ràng, không silent fallback: bad ir_version, missing op, unsupported isa (chỉ portable/avx2/avx512), tokens 1..16, dim 1..128, quantization fp32/int8/int16, shape dim 0..1000000, layout hợp lệ. Fuzz 200 trials: valid pass, invalid reject, không false accept/reject.

## Implementations

- Python `training/compiler/ir.py`: `build_ir`, `verify_ir`, `ir_empty`.
- Rust `crates/rune-ir`: cùng structs + `verify` + `shape_key` + `kernel_for`.
- C++ `core/compiler/ir.h`: cùng structs + `verifyIr` + `shapeKey` + `kernelFor`.

Ba bên cùng kernel ids, shapes, fusion groups. Python là reference, Rust là second implementation, C++ chỉ consume plan (không duplicate compiler logic).
