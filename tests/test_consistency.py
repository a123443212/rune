# RUNE — Relational Unified Neural Evaluator
# Copyright (C) 2026 a123443212
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# This project is dual-licensed under the MIT License and the
# Apache License, Version 2.0. You may choose either license
# when using, copying, modifying, or distributing this software.
#
# MIT License: https://opensource.org/license/mit
# Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, this
# software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied.

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.conftest import find_binding

_d = find_binding()
if _d and _d not in sys.path:
    sys.path.insert(0, _d)

import pytest

try:
    import rune_bindings as rb

    HAS_BINDINGS = True
except ImportError:
    HAS_BINDINGS = False

from training.export.export import load_exported_arrays, read_header
from training.export.export import export_model
from training.export.quantize import fake_quantize, quantization_report
from training.models.rune_models import EXPORT_ORDER, build_model

needs_bindings = pytest.mark.skipif(not HAS_BINDINGS, reason="bindings not built")


def copy_weights_to_cpp(torch_model, cpp_model):
    arch_t = torch_model.arch_tensors()
    emb = torch_model.embedding_tensors()
    for g in range(9):
        cpp_model.set_embedding(g, [float(x) for x in emb[f"emb{g}"].numpy().reshape(-1)])
    names = list(EXPORT_ORDER[torch_model.arch_id])
    flat = []
    for n in names:
        flat.extend([float(x) for x in np.asarray(arch_t[n]).reshape(-1)])
    cpp_model.set_arch_tensors(names, flat)


@needs_bindings
def test_embedder_matches_cpp_tokens():
    torch.manual_seed(3)
    for arch in ["RUNE-MLP", "RUNE-ATTN-GAB", "RUNE-ATTN-SOFT", "RUNE-SFNN"]:
        tm = build_model(arch)
        tm.eval()
        cm = rb.RuneModel(arch)
        copy_weights_to_cpp(tm, cm)
        fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        cpp_tok = np.array(cm.tokens_for_fen(fen))
        feats = __import__("training.features.python_features", fromlist=["extract_features"]).extract_features(fen)
        toks = []
        emb = tm.embedder
        with torch.no_grad():
            for g in range(9):
                idx = [i for gg, i in feats if gg == g]
                e = emb.tables[g](torch.tensor(idx)).sum(0).clamp(0, 1).numpy()
                toks.append(e)
            toks[0] = toks[0] + toks[8]
            toks = toks[:8]
        py_tok = np.concatenate(toks)
        assert np.allclose(cpp_tok, py_tok, atol=1e-5), arch


@needs_bindings
def test_full_model_matches_cpp_eval():
    torch.manual_seed(4)
    for arch in ["RUNE-MLP", "RUNE-ATTN-GAB", "RUNE-ATTN-SOFT", "RUNE-SFNN"]:
        tm = build_model(arch)
        tm.eval()
        cm = rb.RuneModel(arch)
        copy_weights_to_cpp(tm, cm)
        fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ]
        for fen in fens:
            cv, cw = cm.eval_fen(fen)
            feats = __import__("training.features.python_features", fromlist=["extract_features"]).extract_features(fen)
            ids, masks = [], []
            for g in range(9):
                idx = [i for gg, i in feats if gg == g]
                ids.append(torch.tensor([idx]))
                masks.append(torch.ones(1, max(1, len(idx))))
                if not idx:
                    ids[-1] = torch.zeros(1, 1, dtype=torch.long)
                    masks[-1] = torch.zeros(1, 1)
            with torch.no_grad():
                pv, pw = tm(ids, masks)
            assert abs(cv - pv.item()) < 1e-4, (arch, fen)
            assert np.allclose(np.array(cw), pw[0].numpy(), atol=1e-4), (arch, fen)


def test_export_roundtrip_fp32(tmp_path):
    torch.manual_seed(5)
    model = build_model("RUNE-ATTN-GAB")
    p = str(tmp_path / "m.rune")
    header = export_model(model, p, quantization="fp32")
    h2, arrays = load_exported_arrays(p)
    assert h2["arch"] == "RUNE-ATTN-GAB"
    assert h2["quantization"] == "fp32"
    arch_t = model.arch_tensors()
    for name in EXPORT_ORDER["RUNE-ATTN-GAB"]:
        assert np.allclose(arrays[name], np.asarray(arch_t[name]), atol=0), name
    for g in range(9):
        emb = model.embedding_tensors()[f"emb{g}"].numpy()
        assert np.allclose(arrays[f"emb{g}"], emb, atol=0)


def test_export_int8_and_fake_quant(tmp_path):
    torch.manual_seed(6)
    model = build_model("RUNE-MLP")
    p = str(tmp_path / "m8.rune")
    export_model(model, p, quantization="int8")
    header, arrays = load_exported_arrays(p)
    assert header["quantization"] == "int8"
    assert arrays["emb0"].dtype == np.int8
    assert arrays["w1"].dtype == np.float32
    for g in range(9):
        scale = header["scales"][f"emb{g}"]
        deq = arrays[f"emb{g}"].astype(np.float32) * scale
        orig = model.embedding_tensors()[f"emb{g}"].numpy()
        assert np.abs(deq - orig).max() < 0.005
    _, arrs = load_exported_arrays(p)
    rep = quantization_report({k: v.astype(np.float32) for k, v in arrs.items() if not k.startswith("emb")})
    assert rep["w1"]["max_abs_err"] >= 0.0
    fq, _ = fake_quantize(np.array([0.0, 0.5, -0.5, 2.0]))
    assert fq.shape == (4,)
