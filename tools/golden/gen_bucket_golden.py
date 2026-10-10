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
from training.features import python_features as pf
from training.models.rune_models import RuneFullModel

OUT_MODELS = os.path.join(os.path.dirname(__file__), "..", "..", "spec", "test-vectors", "models")
OUT_V10 = os.path.join(os.path.dirname(__file__), "..", "..", "spec", "test-vectors", "v10")

SEED = 23


def batch_single(feats):
    ids = []
    masks = []
    for g in range(pf.NUM_GROUPS):
        idxs = [i for (gg, i) in feats if gg == g]
        n = len(idxs)
        if n == 0:
            idxs = [0]
        ids.append(torch.tensor([idxs], dtype=torch.long))
        m = torch.zeros(1, len(idxs))
        m[0, :n] = 1.0
        masks.append(m)
    return ids, masks


def knight_fen(n):
    banned = {(2, 6), (6, 6), (3, 5), (5, 5)}
    squares = []
    for r in range(7, -1, -1):
        for f in range(8):
            if (f, r) in ((4, 0), (4, 7)):
                continue
            if (f, r) in banned:
                continue
            squares.append((f, r))
    assert n <= len(squares)
    grid = [[None] * 8 for _ in range(8)]
    grid[0][4] = "K"
    grid[7][4] = "k"
    for f, r in squares[:n]:
        grid[r][f] = "N"
    rows = []
    for r in range(7, -1, -1):
        row = ""
        empty = 0
        for f in range(8):
            c = grid[r][f]
            if c is None:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += c
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows) + " w - - 0 1"


def main():
    os.makedirs(OUT_MODELS, exist_ok=True)
    os.makedirs(OUT_V10, exist_ok=True)
    torch.manual_seed(SEED)
    model = RuneFullModel("RUNE-ATTN-GAB", buckets=3)
    model.eval()
    path = os.path.join(OUT_MODELS, "small-gab-b3-fp32.rune")
    header = export_model(model, path, quantization="fp32")
    assert header["head_buckets"] == 3
    names = [t["name"] for t in header["tensors"]]
    assert "emb8" in names
    assert "w1_b0" in names and "w1_b2" in names and "bwdl_b1" in names
    assert "w1" not in names

    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        knight_fen(12),
        knight_fen(11),
        knight_fen(6),
        knight_fen(5),
    ]
    vectors = []
    with torch.no_grad():
        for fen in fens:
            feats = pf.extract_features(fen)
            board, _, _, _ = pf.parse_fen(fen)
            phase = pf.game_phase(board)
            ids, masks = batch_single(feats)
            value, wdl = model(ids, masks)
            got_phase = model.head.phases_from_ids(ids, masks).item()
            assert got_phase == phase, (fen, got_phase, phase)
            vectors.append({
                "fen": fen,
                "phase": phase,
                "value": float(value.item()),
                "wdl": [float(x) for x in wdl[0].tolist()],
            })
    phases = sorted(v["phase"] for v in vectors)
    assert phases == [0, 0, 0, 1, 1, 2, 2], phases
    by_fen = {v["fen"]: v["phase"] for v in vectors}
    assert by_fen[knight_fen(12)] == 0
    assert by_fen[knight_fen(11)] == 1
    assert by_fen[knight_fen(6)] == 1
    assert by_fen[knight_fen(5)] == 2
    doc = {
        "feature_version": pf.FEATURE_VERSION,
        "model": "small-gab-b3-fp32.rune",
        "seed": SEED,
        "vectors": vectors,
    }
    with open(os.path.join(OUT_V10, "bucket_heads.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote bucket_heads.json with %d vectors" % len(vectors))


if __name__ == "__main__":
    main()
