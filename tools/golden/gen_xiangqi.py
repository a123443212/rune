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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from training.export.export import export_model
from training.games import get as get_game
from training.models.rune_models import build_model

OUT_MODELS = os.path.join(os.path.dirname(__file__), "..", "..", "spec", "test-vectors", "models")
OUT_XIANGQI = os.path.join(os.path.dirname(__file__), "..", "..", "spec", "test-vectors", "xiangqi")

FENS = [
    "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1",
    "2eak4/4k4/2h1a4/p3p1p1p/4c4/4P4/P1P1C1P1P/9/9/RHEAKAEHR w - - 0 1",
    "4k4/4a4/9/9/4r4/9/9/9/9/4K4 w - - 0 1",
    "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR b - - 0 1",
    "4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1",
]


def feats_to_tensors(g, feats):
    ids = []
    masks = []
    for gg in range(g.num_groups):
        idx = [i for q, i in feats if q == gg]
        if idx:
            ids.append(torch.tensor([idx]))
            masks.append(torch.ones(1, len(idx)))
        else:
            ids.append(torch.zeros(1, 1, dtype=torch.long))
            masks.append(torch.zeros(1, 1))
    return ids, masks


def build():
    os.makedirs(OUT_MODELS, exist_ok=True)
    os.makedirs(OUT_XIANGQI, exist_ok=True)
    g = get_game("xiangqi")
    torch.manual_seed(29)
    m = build_model("RUNE-MLP", game="xiangqi")
    m.eval()
    export_model(m, os.path.join(OUT_MODELS, "xiangqi-mlp-fp32.rune"), quantization="fp32")
    vectors = []
    for fen in FENS:
        feats = g.extract(fen)
        ids, masks = feats_to_tensors(g, feats)
        with torch.no_grad():
            v, w = m(ids, masks)
        vectors.append({
            "fen": fen,
            "features": [list(f) for f in feats],
            "phase": g.phase(fen),
            "context": g.context(fen),
            "value": float(v[0]),
            "wdl": [float(x) for x in w[0]],
        })
    with open(os.path.join(OUT_XIANGQI, "eval.json"), "w") as f:
        json.dump({
            "version": 1,
            "game": "xiangqi",
            "feature_version": g.feature_version,
            "model": "../models/xiangqi-mlp-fp32.rune",
            "vectors": vectors,
        }, f, indent=2)
    for v in vectors:
        print(v["fen"][:40], "feats=%d" % len(v["features"]), "value=%.6f" % v["value"])


if __name__ == "__main__":
    build()
