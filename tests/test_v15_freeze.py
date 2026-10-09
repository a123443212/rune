import os
import re

import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read(path):
    with open(os.path.join(ROOT, path)) as f:
        return f.read()


def _rust_const(path, name):
    src = _read(path)
    m = re.search(r'pub const %s:\s*&str\s*=\s*"([^"]+)"' % name, src)
    assert m, name
    return m.group(1)


def test_feature_version_single_source():
    spec = _read("spec/VERSIONS.md")
    assert "grouped_hkav2_fullthreats_v02" in spec
    assert _rust_const("crates/rune-spec/src/games.rs", "FEATURE_VERSION") == "grouped_hkav2_fullthreats_v02"
    py = _read("training/features/python_features.py")
    assert "grouped_hkav2_fullthreats_v02" in py


def test_model_format_version_single_source():
    assert "currently 2" in _read("spec/VERSIONS.md")
    assert "MODEL_FORMAT_VERSION: u32 = 2" in _read("crates/rune-spec/src/header.rs")


def test_interaction_graph_version_three_way():
    assert _rust_const("crates/rune-ir/src/incremental.rs", "GRAPH_VERSION") == "v13-graph-01"
    assert _rust_const("crates/rune-ir/src/incremental.rs", "INVALIDATION_VERSION") == "v13-inv-01"
    assert "v13-graph-01" in _read("core/incremental/interaction_graph.cpp")
    assert "v13-inv-01" in _read("core/incremental/interaction_graph.cpp")
    from training.rune_v13.interaction_graph import GRAPH_VERSION, INVALIDATION_VERSION
    assert GRAPH_VERSION == "v13-graph-01"
    assert INVALIDATION_VERSION == "v13-inv-01"


def test_tolerance_consts_match_contract():
    src = _read("crates/rune-spec/src/tolerance.rs")
    for want in ("MATVEC_ABS: f32 = 1e-5", "HEAD_ABS: f32 = 2e-5", "TANH_ABS: f32 = 2e-6",
                 "RESIDUAL_ABS: f32 = 1e-6", "GATE_FMA_ABS: f32 = 1e-6"):
        assert want in src
    contract = _read("spec/numerical-contract.md")
    for want in ("1e-5", "2e-5", "2e-6", "1e-6"):
        assert want in contract


def test_final_candidate_arch_ids_stable():
    from training.models.rune_models import build_model
    assert build_model("RUNE-ATTN-GAB").model_spec()["arch"] == "RUNE-ATTN-GAB"
    assert build_model("RUNE-MLP").model_spec()["arch"] == "RUNE-MLP"
    assert "RUNE-ATTN-GAB" in _read("spec/architecture/runtime.md")
    assert "RUNE-MLP" in _read("spec/architecture/runtime.md")


def test_quant_helpers_single_semantics():
    from training.rune_v13.quant_delta import dequant_tokens
    acc = torch.zeros(8, 32, dtype=torch.int32)
    toks = dequant_tokens(acc, torch.full((8,), 0.01))
    assert bool(((toks >= 0.0) & (toks <= 1.0)).all())
