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
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from training.experiments.screening import ScreeningRunner, load_config
from tests.conftest import find_binding

_d = find_binding()
if _d and _d not in sys.path:
    sys.path.insert(0, _d)

try:
    import rune_bindings as rb
    HAS_BINDINGS = True
except ImportError:
    HAS_BINDINGS = False
    rb = None

pytestmark = pytest.mark.skipif(not HAS_BINDINGS, reason="bindings not built")


def make_pool(games=12, seed=0):
    import random

    rng = random.Random(seed)
    pool = []
    for g in range(games):
        b = rb.Board()
        ply = 0
        while ply < 120:
            moves = b.legal_moves()
            if not moves:
                break
            pool.append({"fen": b.to_fen(), "value": 0.05 * ((ply % 5) - 2), "wdl": 1,
                         "game_id": f"sg{g}", "ply": ply, "teacher_value": 0.0,
                         "student_value": 0.0})
            m = moves[rng.randrange(len(moves))]
            mv = rb.Move()
            mv.from_sq, mv.to_sq, mv.promo = m[0], m[1], m[2]
            b.make_move(mv)
            ply += 1
    return pool


def base_cfg(tmp, milestones=(64, 128)):
    return {
        "experiment": {"name": "smoke", "seed": 0},
        "out_dir": str(tmp / "runs"),
        "research": {"allow_experimental": True},
        "data": {"dataset": "smoke", "teacher": "smoke_t", "positions": list(milestones)},
        "models": ["rune_mlp", "rune_attn_gab"],
        "training": {"lr": 1e-3, "batch_size": 8, "log_every": 1000},
        "loss": {"value": True, "wdl": True, "ranking": False},
    }


def test_screening_runner_checkpoints_per_milestone(tmp_path):
    runner = ScreeningRunner(base_cfg(tmp_path))
    out = runner.run(make_pool())
    for model in ("rune_mlp", "rune_attn_gab"):
        for tag in ("64", "128"):
            d = os.path.join(out, model, tag)
            assert os.path.exists(os.path.join(d, "model.pt"))
            assert os.path.exists(os.path.join(d, "meta.json"))
            with open(os.path.join(d, "metrics.json")) as f:
                m = json.load(f)
            assert m["trained_positions"] >= int(tag)
            assert m["data_info"]["dataset_hash"]
            assert m["git_commit"]
    out2 = ScreeningRunner(base_cfg(tmp_path)).run(make_pool())
    assert out2 == out


def test_report_plot_promote(tmp_path):
    runner = ScreeningRunner(base_cfg(tmp_path))
    runs = runner.run(make_pool())
    rep = str(tmp_path / "reports")
    r = subprocess.run([sys.executable, "tools/screening/report.py", "--runs", runs,
                        "--out-dir", rep], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert os.path.exists(os.path.join(rep, "comparison_64.md"))
    with open(os.path.join(rep, "comparison_64.md")) as f:
        body = f.read()
    assert "rune_mlp" in body and "rune_attn_gab" in body
    assert "composite" in body.lower() or "No composite" in body
    r = subprocess.run([sys.executable, "tools/screening/plot.py", "--runs", runs,
                        "--out-dir", rep], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert os.path.exists(os.path.join(rep, "val_loss.png"))
    r = subprocess.run([sys.executable, "tools/screening/promote.py", "--runs", runs],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "human decision" in r.stdout
    out_cfg = str(tmp_path / "p250.json")
    r = subprocess.run([sys.executable, "tools/screening/promote.py", "--runs", runs,
                        "--write-250m", out_cfg, "--model", "rune_mlp"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with open(out_cfg) as f:
        cfg = json.load(f)
    assert cfg["data"]["positions"] == [250_000_000]
    assert cfg["models"] == ["rune_mlp"]


def test_experimental_guard(tmp_path):
    import pytest

    cfg = base_cfg(tmp_path)
    cfg.pop("research")
    cfg["models"] = ["rune_rel_d"]
    with pytest.raises(ValueError, match="without opt-in"):
        ScreeningRunner(cfg)
    cfg["research"] = {"allow_experimental": True}
    assert ScreeningRunner(cfg).exp_name == "smoke"
    cfg2 = base_cfg(tmp_path)
    cfg2.pop("research")
    cfg2["models"] = ["rune_mlp"]
    assert ScreeningRunner(cfg2).exp_name == "smoke"


def test_load_config_yaml_and_json(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("experiment:\n  name: x\n  seed: 1\n")
    assert load_config(str(p))["experiment"]["seed"] == 1
    q = tmp_path / "c.json"
    q.write_text('{"experiment": {"seed": 2}}')
    assert load_config(str(q))["experiment"]["seed"] == 2
