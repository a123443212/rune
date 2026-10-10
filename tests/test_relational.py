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

from training.features.context import context_vector
from training.features.python_features import extract_features
from training.models.relational import build_rel_model

needs_bindings = pytest.mark.skipif(not HAS_BINDINGS, reason="bindings not built")

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]


def copy_to_cpp(tm, cm):
    emb = tm.embedding_tensors()
    for g in range(9):
        cm.set_embedding(g, [float(x) for x in emb[f"emb{g}"].numpy().reshape(-1)])
    arch = tm.arch_tensors()
    names = tm.export_order()
    flat = []
    for n in names:
        flat.extend([float(x) for x in np.asarray(arch[n]).reshape(-1)])
    cm.set_arch_tensors(names, flat)


def batch_for(tm, fen):
    from training.features.python_features import VOCAB_SIZES

    feats = extract_features(fen)
    ids, masks = [], []
    for g in range(9):
        idx = [i for gg, i in feats if gg == g]
        ids.append(torch.tensor([idx if idx else [0]]))
        masks.append(torch.tensor([[1.0] * len(idx) if idx else [0.0]]))
    ctx = torch.tensor([context_vector(fen)], dtype=torch.float32)
    return ids, masks, ctx


@needs_bindings
def test_context_parity():
    for fen in FENS:
        cm = rb.FlexModel(8, 32, "clip", 1.0, True)
        assert np.allclose(cm.context_for_fen(fen), context_vector(fen), atol=1e-6)


@needs_bindings
def test_flex_tokens_parity_all_layouts():
    torch.manual_seed(11)
    for tokens, dim in [(8, 32), (6, 32), (10, 32), (8, 24), (8, 40), (6, 24), (10, 40)]:
        tm = build_rel_model(tokens=tokens, dim=dim)
        tm.eval()
        cm = rb.FlexModel(tokens, dim, "clip", 1.0, False)
        copy_to_cpp(tm, cm)
        for fen in FENS:
            cpp_tok = np.array(cm.tokens_for_fen(fen))
            ids, masks, _ = batch_for(tm, fen)
            with torch.no_grad():
                py_tok = tm.embedder(ids, masks)[0].numpy().reshape(-1)
            assert np.allclose(cpp_tok, py_tok, atol=1e-5), (tokens, dim, fen)


@needs_bindings
def test_relational_eval_parity():
    torch.manual_seed(12)
    variants = [
        (8, 32, "clip", 1.0, False),
        (8, 32, "hard_sigmoid", 0.5, False),
        (8, 32, "clip", 1.0, True),
        (6, 32, "clip", 0.75, True),
        (10, 40, "hard_sigmoid", 1.0, True),
    ]
    for tokens, dim, gate, alpha, dyn in variants:
        tm = build_rel_model(tokens=tokens, dim=dim, gate=gate, alpha=alpha, dynamic_bias=dyn)
        tm.eval()
        cm = rb.FlexModel(tokens, dim, gate, alpha, dyn)
        copy_to_cpp(tm, cm)
        for fen in FENS:
            cv, cw = cm.eval_fen(fen)
            ids, masks, ctx = batch_for(tm, fen)
            with torch.no_grad():
                pv, pw = tm(ids, masks, ctx)
            assert abs(cv - pv.item()) < 1e-4, (tokens, dim, gate, alpha, dyn, fen)
            assert np.allclose(np.array(cw), pw[0].numpy(), atol=1e-4)


def test_rel_param_accounting():
    from training.features.context import CONTEXT_DIM
    a = build_rel_model(8, 32, dynamic_bias=False)
    b = build_rel_model(8, 32, dynamic_bias=True)
    assert b.parameter_count() - a.parameter_count() == 2 * 8 * CONTEXT_DIM
    small = build_rel_model(6, 24)
    big = build_rel_model(10, 40)
    assert small.parameter_count() < a.parameter_count() < big.parameter_count()
    assert a.model_spec()["geometric_bias"] == "static"
    assert b.model_spec()["geometric_bias"] == "dynamic"
    assert b.model_spec()["gate"] == "clip"


def test_context_us_them_split():
    from training.features.context import CONTEXT_DIM, CONTEXT_NAMES

    assert CONTEXT_DIM == 17
    assert len(CONTEXT_NAMES) == 17
    start = context_vector("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert len(start) == 17
    assert start[2] == start[3] == 1.0
    assert start[4] == start[5] == 0.5
    assert start[6] == start[7] == 0.5
    assert start[8] == start[9] == 0.5
    assert start[10] == start[11] == 1.0
    end = context_vector("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
    assert end[2] == end[3] == 0.375
    assert end[6] == end[7] == 0.25
    assert end[10] == end[11] == 0.3125
    assert end[1] == 1.0


def test_rel_export_header(tmp_path):
    from training.export.export import export_model, load_exported_arrays

    torch.manual_seed(13)
    model = build_rel_model(tokens=6, dim=24, gate="hard_sigmoid", alpha=0.5, dynamic_bias=True)
    p = str(tmp_path / "rel.rune")
    export_model(model, p, quantization="int16")
    header, arrays = load_exported_arrays(p)
    assert header["arch"] == "RUNE-REL-02"
    assert header["gate"] == "hard_sigmoid"
    assert header["alpha"] == 0.5
    assert header["context_dim"] == 17
    assert header["tensors"][0]["dtype"] == "int16"
    assert arrays["emb0"].dtype == np.int16
    assert "dynU" in arrays and "dynW" in arrays
