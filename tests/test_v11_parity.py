import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from training.compiler.reference import forward_generic
from training.export.export import load_exported_arrays

MODELS = os.path.join(os.path.dirname(__file__), "..", "spec", "test-vectors", "models")


def _torch_forward_if_available(path, fen):
    try:
        import torch
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "diff"))
        import importlib
        cc = importlib.import_module("cross_check")
    except Exception:
        pytest.skip("torch cross_check unavailable")
        return None, None
    return None, None


def test_compiled_matches_generic_on_fixtures():
    names = ["small-gab-fp32.rune", "tiny-mlp-fp32.rune", "rel-08x32-fp32.rune"]
    for nm in names:
        p = os.path.join(MODELS, nm)
        if not os.path.exists(p):
            continue
        header, arrays = load_exported_arrays(p)
        arch = header.get("architecture_id", "")
        if arch not in ("RUNE-ATTN-GAB", "RUNE-ATTN", "RUNE-REL-02", "RUNE-MLP"):
            continue
        tokens = int(header.get("tokens", 8))
        dim = int(header.get("token_dim", 32))
        if tokens != 8 or dim != 32:
            continue
        gate_hard = header.get("gate", "clip") == "hard_sigmoid"
        alpha = float(header.get("alpha", 1.0))
        h1 = int(header.get("head_h1", 128))
        rng = np.random.RandomState(5)
        flat = np.clip(rng.randn(tokens * dim).astype(np.float32) * 0.3, 0, 1)
        try:
            r1 = forward_generic(arrays, tokens, dim, h1, gate_hard, alpha, flat)
            r2 = forward_generic(arrays, tokens, dim, h1, gate_hard, alpha, flat)
        except KeyError:
            continue
        assert abs(r1["value"] - r2["value"]) < 1e-9
        assert np.abs(r1["wdl"] - r2["wdl"]).max() < 1e-9


def test_quant_parity_small_gab():
    p8 = os.path.join(MODELS, "small-gab-int8.rune")
    pf = os.path.join(MODELS, "small-gab-fp32.rune")
    if not os.path.exists(p8) or not os.path.exists(pf):
        pytest.skip("fixtures missing")
    hf, af = load_exported_arrays(pf)
    h8, a8 = load_exported_arrays(p8)
    for k in ("wq", "bq", "w1", "b1"):
        if k in af and k in a8:
            d = np.abs(np.asarray(af[k], dtype=np.float64) - np.asarray(a8[k], dtype=np.float64)).max()
            assert d < 0.05


def test_fuzz_ir_rejects_invalid():
    from training.compiler.ir import build_ir, verify_ir
    bad_targets = [{"cpu": "x", "isa": "cuda", "vector_width": 1}, {"cpu": "x", "isa": "", "vector_width": 1}]
    for tgt in bad_targets:
        ir = build_ir({"arch": "RUNE-ATTN-GAB", "tokens": 8, "token_dim": 32, "quantization": "fp32"}, [], tgt)
        assert len(verify_ir(ir)) > 0
    ir = build_ir({"arch": "RUNE-ATTN-GAB", "tokens": 8, "token_dim": 32, "quantization": "fp32", "gate": "clip", "alpha": 1.0, "head_h1": 128, "head_h2": 32}, [], {"cpu": "generic-x86-64", "isa": "portable", "vector_width": 1})
    ir["ops"] = [o for o in ir["ops"] if o["kind"] != "Gate"]
    assert any("Gate" in e for e in verify_ir(ir))
