import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from training.compiler.reference import forward_generic
from training.export.export import load_exported_arrays

V11 = os.path.join(os.path.dirname(__file__), "..", "spec", "test-vectors", "v11")


def test_compiled_golden():
    with open(os.path.join(V11, "compiled.json")) as f:
        doc = json.load(f)
    assert doc["ir_version"] == "1.0"
    header, arrays = load_exported_arrays(os.path.join(os.path.dirname(__file__), "..", "spec", "test-vectors", "models", "small-gab-fp32.rune"))
    tokens, dim, h1 = 8, 32, 128
    gate_hard = header.get("gate", "clip") == "hard_sigmoid"
    alpha = float(header.get("alpha", 1.0))
    for v in doc["vectors"]:
        flat = np.asarray(v["input"], dtype=np.float32)
        r = forward_generic(arrays, tokens, dim, h1, gate_hard, alpha, flat)
        assert abs(r["value"] - v["value"]) < 1e-9
        assert np.abs(np.asarray(r["wdl"]) - np.asarray(v["wdl"])).max() < 1e-9
