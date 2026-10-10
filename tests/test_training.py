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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.datasets.rune_dataset import make_loader
from training.experiments.learning_curve import LearningCurve
from training.trainer.trainer import Trainer

FEN_POOL = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbqkb1r/pp2pppp/5n2/2pp4/3P4/2N5/PPP1PPPP/R1BQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
]


def make_records(n=48):
    recs = []
    for i in range(n):
        fen = FEN_POOL[i % len(FEN_POOL)]
        v = 0.1 * ((i % 7) - 3)
        recs.append({
            "fen": fen,
            "value": v,
            "wdl": 1 if abs(v) < 0.15 else (0 if v > 0 else 2),
            "game_id": f"g{i // 4}",
            "ply": 10 + (i % 30),
            "teacher_value": v,
            "student_value": 0.0,
        })
    return recs


def test_dataset_loader_runs():
    recs = make_records()
    loader, ds = make_loader(recs, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    ids, masks, value, wdl = batch
    assert len(ids) == 9 and value.shape[0] == 8


def test_trainer_trains_and_checkpoints(tmp_path):
    recs = make_records()
    cfg = {"arch": "RUNE-MLP", "seed": 0, "lr": 1e-3, "lambda_wdl": 0.5, "lambda_rank": 0.1}
    tr = Trainer(cfg)
    before = tr.evaluate(recs[:16])["total"]
    tr.fit(recs, recs[:16], max_positions=512, batch_size=8, log_every=1000,
           ckpt_dir=str(tmp_path / "ckpt"))
    assert tr.positions_seen >= 512
    assert os.path.exists(str(tmp_path / "ckpt" / "model.pt"))
    assert os.path.exists(str(tmp_path / "ckpt" / "meta.json"))
    after = tr.evaluate(recs[:16])["total"]
    assert after <= before + 1.0


def test_learning_curve_tiny(tmp_path):
    recs = make_records(64)
    cfg = {
        "arch": "RUNE-ATTN-GAB",
        "seed": 0,
        "lr": 1e-3,
        "lambda_wdl": 0.5,
        "lambda_rank": 0.0,
        "batch_size": 8,
        "log_every": 1000,
        "milestones": [100_000_000, 250_000_000],
        "milestone_budgets": {"100000000": 256, "250000000": 256},
        "sampling": "random",
        "max_positions_cap": 256,
    }
    lc = LearningCurve(cfg)
    summary = lc.run(recs, str(tmp_path / "lc"))
    assert len(summary) == 2
    assert os.path.exists(str(tmp_path / "lc" / "learning_curve.csv"))
    assert os.path.exists(str(tmp_path / "lc" / "model_100000000.rune"))
