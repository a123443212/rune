import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from training.compiler.ir import IR_VERSION, build_ir, verify_ir


def test_ir_build_minimal():
    spec = {"arch": "RUNE-ATTN-GAB", "arch_version": "0.2.0", "tokens": 8, "token_dim": 32, "quantization": "fp32", "gate": "clip", "alpha": 1.0, "head_h1": 128, "head_h2": 32}
    metas = [{"name": "wq", "shape": [32, 32], "dtype": "float32"}]
    ir = build_ir(spec, metas, {"cpu": "generic-x86-64", "isa": "portable", "vector_width": 1})
    assert ir["ir_version"] == IR_VERSION
    assert len(ir["ops"]) == 15
    assert verify_ir(ir) == []


def test_ir_rejects_bad_isa():
    spec = {"arch": "RUNE-ATTN-GAB", "tokens": 8, "token_dim": 32, "quantization": "fp32"}
    ir = build_ir(spec, [], {"cpu": "x", "isa": "neon", "vector_width": 4})
    errs = verify_ir(ir)
    assert any("isa" in e for e in errs)


def test_ir_rejects_bad_shape():
    spec = {"arch": "RUNE-ATTN-GAB", "tokens": 64, "token_dim": 32, "quantization": "fp32"}
    ir = build_ir(spec, [], {"cpu": "x", "isa": "portable", "vector_width": 1})
    errs = verify_ir(ir)
    assert any("tokens" in e for e in errs)


def test_ir_rejects_bad_quant():
    spec = {"arch": "RUNE-ATTN-GAB", "tokens": 8, "token_dim": 32, "quantization": "int4"}
    ir = build_ir(spec, [], {"cpu": "x", "isa": "portable", "vector_width": 1})
    errs = verify_ir(ir)
    assert any("quantization" in e for e in errs)


def test_ir_requires_bias():
    spec = {"arch": "RUNE-ATTN-GAB", "tokens": 8, "token_dim": 32, "quantization": "fp32"}
    ir = build_ir(spec, [], {"cpu": "x", "isa": "portable", "vector_width": 1})
    ir["ops"] = [o for o in ir["ops"] if o.get("kind") != "Bias"]
    errs = verify_ir(ir)
    assert any("Bias" in e for e in errs)


def test_ir_version_independent():
    from training.compiler.ir import SPEC_VERSION
    assert IR_VERSION == "1.1"
    assert SPEC_VERSION == "RUNE-11"
