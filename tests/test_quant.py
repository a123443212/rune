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

try:
    import rune_bindings as rb
    HAS_BINDINGS = True
except ImportError:
    HAS_BINDINGS = False

import pytest
needs_bindings = pytest.mark.skipif(not HAS_BINDINGS, reason="bindings not built")

from training.export.export import export_model, load_exported_arrays
from training.export.quantize import fake_quantize
from training.models.relational import build_rel_model
from training.models.rune_models import build_model


def cpp_eval(path, fen):
    from training.export.export import load_exported_arrays as lea

    header, arrays = lea(path)
    arch = header["arch"]
    if arch == "RUNE-REL-02":
        m = rb.FlexModel(header["tokens"], header["token_dim"], header.get("gate", "clip"),
                         header.get("alpha", 1.0), header["geometric_bias"] == "dynamic")
    else:
        m = rb.RuneModel(arch)
    for g in range(9):
        arr = arrays[f"emb{g}"]
        if str(arr.dtype) in ("int8", "int16"):
            arr = arr.astype("float32") * header["scales"][f"emb{g}"]
        m.set_embedding(g, [float(x) for x in arr.reshape(-1)])
    names = [t["name"] for t in header["tensors"] if not t["name"].startswith("emb")]
    flat = []
    for n in names:
        flat.extend([float(x) for x in arrays[n].reshape(-1)])
    m.set_arch_tensors(names, flat)
    return m.eval_fen(fen)


def test_int16_export_roundtrip(tmp_path):
    torch.manual_seed(21)
    model = build_model("RUNE-MLP")
    p = str(tmp_path / "m16.rune")
    export_model(model, p, quantization="int16")
    header, arrays = load_exported_arrays(p)
    assert arrays["emb0"].dtype == np.int16
    scale = header["scales"]["emb0"]
    deq = arrays["emb0"].astype(np.float32) * scale
    orig = model.embedding_tensors()["emb0"].numpy()
    assert np.abs(deq - orig).max() < 0.001


def test_quant_consistency_all_modes(tmp_path):
    if not HAS_BINDINGS:
        pytest.skip("bindings not built")
    torch.manual_seed(22)
    fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    for arch, kw in [("RUNE-MLP", {}), ("RUNE-REL-02", {"dynamic_bias": True})]:
        model = build_rel_model(**kw) if arch == "RUNE-REL-02" else build_model(arch)
        vals = {}
        for quant in ("fp32", "int16", "int8"):
            p = str(tmp_path / f"{arch}_{quant}.rune")
            export_model(model, p, quantization=quant)
            vals[quant] = cpp_eval(p, fen)[0]
        assert abs(vals["fp32"] - vals["int16"]) < 0.002
        assert abs(vals["fp32"] - vals["int8"]) < 0.05


def test_fake_quantize_bits():
    arr = np.array([0.0, 0.5, -0.5, 2.0], dtype=np.float32)
    fq8, _ = fake_quantize(arr, bits=8)
    fq16, _ = fake_quantize(arr, bits=16)
    assert fq8.shape == (4,) and fq16.shape == (4,)
    e8 = np.abs(arr - fq8).max()
    e16 = np.abs(arr - fq16).max()
    assert e16 <= e8
