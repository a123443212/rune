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

import json
import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.features import python_features as pf
V10 = os.path.join(os.path.dirname(__file__), "..", "spec", "test-vectors", "v10")
def load(name):
    with open(os.path.join(V10, name)) as f:
        return json.load(f)
def test_features_match_vectors():
    data = load("features.json")
    assert data["feature_version"] == pf.FEATURE_VERSION
    for v in data["vectors"]:
        got = pf.extract_features(v["fen"])
        want = [tuple(x) for x in v["features"]]
        assert got == want, v["fen"]
def test_accumulator_tokens_match():
    data = load("accumulator.json")
    assert len(data["vectors"]) > 0
    for v in data["vectors"]:
        tok = v["tokens"]
        for x in tok:
            assert 0.0 <= x <= 1.0
        assert len(tok) == 8 * pf.TOKEN_DIM
def test_quant_half_away():
    data = load("quant.json")
    from tools.golden.gen_vectors import quant_half_away
    for w, q in zip(data["inputs"], data["int8"]):
        assert quant_half_away(w, data["scale"], 127) == q
    for w, q in zip(data["inputs"], data["int16"]):
        assert quant_half_away(w, data["scale"], 32767) == q
def test_mixer_shapes():
    data = load("mixer.json")
    assert len(data["mixed"]) == data["tokens"] * data["dim"]
    assert len(data["q"]) == data["tokens"] * data["dim"]
    assert len(data["scores"]) == data["tokens"] * data["tokens"]
def test_mixer_gates_fire_and_mix():
    import numpy as np
    data = load("mixer.json")
    t, d = data["tokens"], data["dim"]
    x = np.array(data["input"], dtype=np.float32).reshape(t, d)
    wq = np.array(data["wq"], dtype=np.float32).reshape(d, d)
    bq = np.array(data["bq"], dtype=np.float32)
    wk = np.array(data["wk"], dtype=np.float32).reshape(d, d)
    bk = np.array(data["bk"], dtype=np.float32)
    wv = np.array(data["wvv"], dtype=np.float32).reshape(d, d)
    bv = np.array(data["bvv"], dtype=np.float32)
    gab = np.array(data["gab"], dtype=np.float32).reshape(t, t)
    q = x @ wq.T + bq
    k = x @ wk.T + bk
    v = x @ wv.T + bv
    s = q @ k.T + gab
    g = np.clip(s, 0, 1)
    gf = np.asarray(data["gates"], dtype=np.float64)
    assert (g.reshape(-1) > 0).sum() == (gf > 0).sum() > 0
    assert np.abs(g.reshape(-1) - gf).max() < 1e-6
    mixed = x + g @ v
    assert np.abs(mixed.reshape(-1) - np.asarray(data["mixed"])).max() < 1e-5
    assert np.abs(mixed.reshape(-1) - np.asarray(data["input"])).max() > 1e-6
def test_head_value_bounded():
    data = load("head.json")
    assert -1.0 <= data["value"] <= 1.0
    assert len(data["wdl"]) == 3
    assert len(data["h1"]) == 128
    assert len(data["h2"]) == 32
def test_routing_nan_refines():
    data = load("routing.json")
    for v in data["vectors"]:
        if v["score"] == "nan":
            assert v["refine"] is True
def test_model_hash_agreement():
    from training.export.export import read_header, fnv1a
    import struct
    d = os.path.join(os.path.dirname(__file__), "..", "spec", "test-vectors", "models")
    for name in sorted(os.listdir(d)):
        if not name.endswith(".rune"):
            continue
        p = os.path.join(d, name)
        h = read_header(p)
        assert h["format"] == 2
        assert "architecture_id" in h
        assert "feature_version" in h
        assert "model_hash" in h
        assert "tensor_metadata" in h
        with open(p, "rb") as f:
            f.read(4)
            import struct as st
            (n,) = st.unpack("<I", f.read(4))
            f.read(n)
            payload = f.read()
        assert format(fnv1a(payload), "016x") == h["model_hash"]
        assert format(fnv1a(payload), "016x") == h["checksum"]
