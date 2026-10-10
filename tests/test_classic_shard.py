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

def test_classic_shard_xiangqi(tmp_path):
    from training.datasets.classic_shard import move_no_of, shard_pipeline
    from training.engine.search_teacher import distill_states
    from training.games import get as get_game
    from training.games.xiangqi_legal import apply_move, legal_moves
    from training.models.rune_models import build_model
    import torch
    torch.manual_seed(2)
    g = get_game("xiangqi")
    m = build_model("RUNE-MLP", game="xiangqi")
    m.eval()
    s = "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1"
    assert move_no_of("xiangqi", s) == 1
    recs = distill_states(g, m, [s, s], legal_moves, apply_move, simulations=2, seed=1)
    for r in recs:
        r["game"] = "xiangqi"
    out = str(tmp_path / "xq")
    manifest = shard_pipeline("xiangqi", legal_moves, recs, out, num_shards=2)
    assert manifest["positions"] == 1
    assert manifest["validated"] == 2
    assert manifest["game"] == "xiangqi"
    import json
    with open(out + "/manifest.json") as f:
        back = json.load(f)
    assert back["positions"] == 1
    assert len(back["shards"][0]["hash"]) == 16


def test_classic_shard_shogi(tmp_path):
    from training.datasets.classic_shard import move_no_of, shard_pipeline
    from training.engine.search_teacher import distill_states
    from training.games import get as get_game
    from training.games.shogi_legal import apply_move, legal_moves
    from training.models.rune_models import build_model
    import torch
    torch.manual_seed(2)
    g = get_game("shogi")
    m = build_model("RUNE-MLP", game="shogi")
    m.eval()
    s = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    assert move_no_of("shogi", s) == 1
    recs = distill_states(g, m, [s], legal_moves, apply_move, simulations=2, seed=1)
    for r in recs:
        r["game"] = "shogi"
    out = str(tmp_path / "sh")
    manifest = shard_pipeline("shogi", legal_moves, recs, out, num_shards=1)
    assert manifest["positions"] == 1
    assert manifest["game"] == "shogi"


def test_classic_shard_rejects_bad():
    from training.datasets.classic_shard import validate_record
    from training.games.xiangqi_legal import legal_moves
    assert not validate_record(legal_moves, {"game": "xiangqi", "state": "bogus"})
    assert not validate_record(legal_moves, {"game": "xiangqi", "state": "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1", "move": "XXXX"})
