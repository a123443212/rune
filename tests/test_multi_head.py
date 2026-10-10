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
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from training.export.export import read_header
from training.features import python_features as pf
from training.models.multi_head import HEAD_DIM, NUM_HEADS, RuneMultiHeadMixer
from training.models.rune_models import RuneFullModel
from tools.golden.gen_bucket_golden import batch_single
from tools.golden.gen_mh_golden import MIXER_KEYS, SEED

V10 = os.path.join(os.path.dirname(__file__), "..", "spec", "test-vectors", "v10")
MODELS = os.path.join(os.path.dirname(__file__), "..", "spec", "test-vectors", "models")
FIXTURE = os.path.join(MODELS, "small-mh4-fp32.rune")


def test_mixer_geometry():
    assert NUM_HEADS == 4
    assert HEAD_DIM == 8
    assert NUM_HEADS * HEAD_DIM == 32


def test_fixture_header_mh():
    h = read_header(FIXTURE)
    assert h["architecture_id"] == "RUNE-ATTN-MH4"
    assert h["attention"] == "multi_head"
    assert h["head_buckets"] == 1
    names = [t["name"] for t in h["tensors"]]
    for hh in range(4):
        for k in MIXER_KEYS:
            assert f"{k}_h{hh}" in names
    assert "wo" in names and "bwo" in names
    assert "w1" in names
    assert "wq" not in names


def test_golden_matches_model():
    with open(os.path.join(V10, "multi_head.json")) as f:
        doc = json.load(f)
    assert doc["feature_version"] == pf.FEATURE_VERSION
    torch.manual_seed(SEED)
    model = RuneFullModel("RUNE-ATTN-MH4", buckets=1)
    model.eval()
    with torch.no_grad():
        for v in doc["vectors"]:
            feats = pf.extract_features(v["fen"])
            ids, masks = batch_single(feats)
            value, wdl = model(ids, masks)
            assert abs(float(value.item()) - v["value"]) < 1e-6, v["fen"]
            for got, want in zip(wdl[0].tolist(), v["wdl"]):
                assert abs(got - want) < 1e-6, v["fen"]


def test_per_head_gab_isolation():
    torch.manual_seed(SEED)
    mixer = RuneMultiHeadMixer()
    mixer.eval()
    x = torch.randn(2, 8, 32) * 0.1
    with torch.no_grad():
        base = mixer(x)
        with_gab = mixer(x)
        assert torch.equal(base, with_gab)
        mixer.gab[2].fill_(0.5)
        changed = mixer(x)
        assert not torch.equal(base, changed)
        per_head = []
        for h in range(4):
            q = mixer.q[h](x)
            k = mixer.k[h](x)
            v = mixer.v[h](x)
            s = q @ k.transpose(-1, -2) + mixer.gab[h]
            per_head.append(torch.clamp(s, 0.0, 1.0) @ v)
        expect = x + mixer.wo(torch.cat(per_head, dim=-1))
        assert torch.allclose(changed, expect, atol=1e-6)
