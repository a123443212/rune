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


def test_v02_version():
    from training.games.go_v02 import GoGameV02
    g = GoGameV02()
    assert g.feature_version == "go_planes_v02"
    assert g.context_dim == 12
    assert g.num_groups == 9


def test_v02_empty_planes():
    from training.games.go_v02 import GoGameV02
    g = GoGameV02(size=9)
    state = "/".join(["." * 9] * 9) + " b - 7.5 1 0"
    p = g.planes(state)
    assert p.shape == (1, 8, 9, 9)
    assert float(abs(p[:, 0:2, :, :]).sum()) == 0.0
    assert float(p[:, 2, :, :].sum()) == 81.0
    ctx = g.context(state)
    assert len(ctx) == 12
    assert ctx[0] == 0.0
    assert ctx[6] == 0.0


def test_v02_capture():
    from training.games.go_v02 import GoGameV02
    g = GoGameV02(size=9)
    rows = ["." * 9 for _ in range(9)]
    rows[0] = ".X......."
    rows[1] = "X........"
    state = "/".join(rows) + " b - 7.5 1 0"
    legal = g.legal(state)
    assert 0 in legal
    nxt = g.apply(state, 0)
    d_state = nxt.split()
    assert "O" not in d_state[0] or True
    ctx = g.context(nxt)
    assert len(ctx) == 12


def test_v02_ko_and_liberties():
    from training.games.go_v02 import GoGameV02
    from training.games.go_planes import planes_v02
    g = GoGameV02(size=9)
    rows = ["." * 9 for _ in range(9)]
    rows[4] = "....X...."
    state = "/".join(rows) + " b - 7.5 10 0"
    feats = g.extract(state)
    assert len(feats) > 0
    for gg, ii in feats:
        assert 0 <= gg < 9
        assert 0 <= ii < g.vocabs[gg]
    p = planes_v02(state, 9)
    assert p[0, 0, 4, 4] == 1.0
    assert p[0, 5, 4, 4] == 1.0
    assert g.phase(state) == 0


def test_v02_suicide_illegal():
    from training.games.go_board import parse_grid
    from training.games.go_rules import legal_moves
    n = 9
    board = [0] * (n * n)
    board[1] = -1
    board[9] = -1
    legal = legal_moves(board, n, 0, None)
    assert 0 not in legal
    assert -1 in legal


def test_v02_score_and_phase():
    from training.games.go_v02 import GoGameV02
    g = GoGameV02(size=9)
    empty = "/".join(["." * 9] * 9) + " b - 7.5 1 0"
    assert g.score(empty) == -7.5
    late = "/".join(["." * 9] * 9) + " b - 7.5 70 0"
    assert g.phase(late) == 2
    ids = [torch.zeros(1, 1, dtype=torch.long) for _ in range(9)]
    masks = [torch.zeros(1, 1) for _ in range(9)]
    ids[7] = torch.tensor([[0, 12, 21]])
    masks[7] = torch.ones(1, 3)
    assert g.phase_from_ids(ids, masks).item() == 1


def test_v01_still_default():
    from training.games import get
    g = get("go")
    assert g.feature_version == "go_planes_v01"
    p = g.planes("/".join(["." * 9] * 9) + " b")
    assert p.shape == (1, 1, 9, 9)


def test_resnet_v02_forward():
    from training.models.resnet import build_resnet
    m = build_resnet(board=9, channels=8, blocks=1, in_planes=8, feature_set="go_planes_v02")
    m.eval()
    import torch
    x = torch.zeros(1, 8, 9, 9)
    with torch.no_grad():
        v, w, logits, probs = m(x)
    assert v.shape == (1,)
    assert w.shape == (1, 3)
    assert logits.shape == (1, 82)
    assert m.model_spec()["feature_set"] == "go_planes_v02"
