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

import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from training.export.export import read_header
from training.features import python_features as pf
from training.models.rune_models import RuneFullModel
from tools.golden.gen_bucket_golden import batch_single, knight_fen, SEED

V10 = os.path.join(os.path.dirname(__file__), "..", "spec", "test-vectors", "v10")
MODELS = os.path.join(os.path.dirname(__file__), "..", "spec", "test-vectors", "models")
FIXTURE = os.path.join(MODELS, "small-gab-b3-fp32.rune")


def load_golden():
    with open(os.path.join(V10, "bucket_heads.json")) as f:
        return json.load(f)


def test_fixture_header_bucketed():
    h = read_header(FIXTURE)
    assert h["head_buckets"] == 3
    names = [t["name"] for t in h["tensors"]]
    assert "emb8" in names
    assert "w1" not in names
    for b in range(3):
        for k in ("w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"):
            assert f"{k}_b{b}" in names


def test_golden_matches_model():
    doc = load_golden()
    assert doc["feature_version"] == pf.FEATURE_VERSION
    assert doc["model"] == "small-gab-b3-fp32.rune"
    torch.manual_seed(SEED)
    model = RuneFullModel("RUNE-ATTN-GAB", buckets=3)
    model.eval()
    with torch.no_grad():
        for v in doc["vectors"]:
            feats = pf.extract_features(v["fen"])
            ids, masks = batch_single(feats)
            value, wdl = model(ids, masks)
            assert abs(float(value.item()) - v["value"]) < 1e-6, v["fen"]
            for got, want in zip(wdl[0].tolist(), v["wdl"]):
                assert abs(got - want) < 1e-6, v["fen"]


def test_phases_from_ids_boundaries():
    torch.manual_seed(SEED)
    model = RuneFullModel("RUNE-ATTN-GAB", buckets=3)
    model.eval()
    cases = [(knight_fen(12), 0), (knight_fen(11), 1), (knight_fen(6), 1), (knight_fen(5), 2)]
    with torch.no_grad():
        for fen, want in cases:
            feats = pf.extract_features(fen)
            board, _, _, _ = pf.parse_fen(fen)
            assert pf.game_phase(board) == want
            ids, masks = batch_single(feats)
            got = model.head.phases_from_ids(ids, masks).item()
            assert got == want, (fen, got, want)


def test_identical_heads_phase_invariant():
    torch.manual_seed(SEED)
    model = RuneFullModel("RUNE-ATTN-GAB", buckets=3)
    model.eval()
    with torch.no_grad():
        src = model.head.heads[0]
        for dst in model.head.heads[1:]:
            dst.load_state_dict(copy.deepcopy(src.state_dict()))
        fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            knight_fen(11),
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ]
        flats = []
        for fen in fens:
            feats = pf.extract_features(fen)
            ids, masks = batch_single(feats)
            x = model.embedder(ids, masks)
            x = model.attn(x)
            flats.append(x.reshape(1, -1))
        flat = torch.cat(flats, dim=0)
        phase = torch.tensor([0, 1, 2])
        value, wdl = model.head(flat, phase)
        for i in range(1, 3):
            v0, _ = model.head.heads[0](flat[i : i + 1])
            assert abs(float(value[i].item()) - float(v0.item())) < 1e-6
