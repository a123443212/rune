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

import torch

START_SHOGI = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
START_XIANGQI = "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1"


def test_versions_registered():
    from training.games.shogi_v02 import ShogiGameV02
    from training.games.xiangqi_v02 import XiangqiGameV02
    assert ShogiGameV02().feature_version == "shogi_sem_v02"
    assert XiangqiGameV02().feature_version == "xiangqi_sem_v02"


def test_v02_extends_v01_in_bounds():
    from training.games import get as get_game
    from training.games.shogi_v02 import ShogiGameV02
    from training.games.xiangqi_v02 import XiangqiGameV02
    for game_v02, game_id, state in ((ShogiGameV02(), "shogi", START_SHOGI), (XiangqiGameV02(), "xiangqi", START_XIANGQI)):
        g = get_game(game_id)
        base = set(g.extract(state))
        ext = game_v02.extract(state)
        assert base.issubset(set(ext))
        assert ext == sorted(set(ext))
        for gg, ii in ext:
            assert 0 <= gg < game_v02.num_groups
            assert 0 <= ii < game_v02.vocabs[gg]
        assert game_v02.extract(state) == ext
        assert game_v02.phase(state) == g.phase(state)
        assert game_v02.context(state) == g.context(state)


def test_shogi_drop_signals():
    from training.games.shogi import parse_sfen
    from training.games.shogi_v02 import ShogiGameV02, checking_drops, drop_mobility, ring_pressure
    g = ShogiGameV02()
    feats = g.extract(START_SHOGI)
    assert (7, 20) in feats
    assert (8, 0) not in feats
    s = "4k4/9/9/9/9/9/9/9/4K4 b P 1"
    board, stm, hand, _ = parse_sfen(s)
    drops = checking_drops(board, hand, stm)
    assert len(drops) > 0
    feats2 = g.extract(s)
    assert (8, 0) in feats2
    assert (7, 20 + min(len(drops), 9)) in feats2
    assert ring_pressure(board, stm) >= 0
    assert drop_mobility(board, hand, stm) >= 0


def test_xiangqi_double_and_mobility():
    from training.games.xiangqi_v02 import XiangqiGameV02, double_attacks, king_mobility
    from training.games.xiangqi import parse_fen
    g = XiangqiGameV02()
    feats = g.extract(START_XIANGQI)
    assert (7, 29 + 0) in feats or (7, 29) in feats
    s = "3aka3/2H1P4/9/4R4/9/9/9/9/9/4K4 b - - 0 1"
    board, stm, _ = parse_fen(s)
    assert len(double_attacks(board, stm)) >= 0
    assert king_mobility(board, stm) == 0
    feats2 = g.extract(s)
    assert (7, 39) in feats2


def test_v02_shared_stack_and_phase_ids():
    from training.games.shogi_v02 import ShogiGameV02
    from training.games.xiangqi_v02 import XiangqiGameV02
    from training.models.rune_models import build_model
    for game_v02, arch_game, state in ((ShogiGameV02(), "shogi", START_SHOGI), (XiangqiGameV02(), "xiangqi", START_XIANGQI)):
        m = build_model("RUNE-MLP", game=arch_game, buckets=3)
        m.eval()
        feats = game_v02.extract(state)
        ids = []
        masks = []
        for gg in range(game_v02.num_groups):
            idx = [i for q, i in feats if q == gg]
            if idx:
                ids.append(torch.tensor([idx]))
                masks.append(torch.ones(1, len(idx)))
            else:
                ids.append(torch.zeros(1, 1, dtype=torch.long))
                masks.append(torch.zeros(1, 1))
        with torch.no_grad():
            v, w = m(ids, masks)
        assert v.shape == (1,)
        assert m.head.phases_from_ids(ids, masks).item() == game_v02.phase(state)
        assert game_v02.phase_from_ids(ids, masks).item() == game_v02.phase(state)
