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

import pytest

START = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


def test_startpos_counts():
    from training.games.shogi_legal import apply_move, legal_moves
    black = legal_moves(START)
    assert len(black) == 30
    assert black == sorted(black)
    white = legal_moves(apply_move(START, black[0]))
    assert len(white) == 30


def test_apply_roundtrip():
    from training.games.shogi import parse_sfen
    from training.games.shogi_legal import apply_move, decode_move, legal_moves, serialize_sfen
    moves = legal_moves(START)
    fr, to, promo = decode_move(moves[0])
    assert 0 <= fr < 81
    assert 0 <= to < 81
    nxt = apply_move(START, moves[0])
    board, stm, hand, move_no = parse_sfen(nxt)
    assert stm == 1
    assert move_no == 2
    assert board[to] is not None
    assert board[fr] is None
    back = serialize_sfen(board, stm, hand, move_no)
    assert parse_sfen(back)[0] == board
    with pytest.raises(ValueError):
        apply_move(START, "D99p")
    with pytest.raises(ValueError):
        apply_move(START, "8080")


def test_must_promote():
    from training.games.shogi_legal import legal_moves
    s = "9/1P7/9/9/9/9/9/9/4K4 b - 1"
    moves = legal_moves(s)
    assert "1001+" in moves
    assert "1001" not in moves


def test_optional_promote():
    from training.games.shogi_legal import legal_moves
    s = "9/9/9/4P4/9/9/9/9/4K4 b - 1"
    moves = legal_moves(s)
    assert "3122" in moves
    assert "3122+" in moves


def test_drop_nifu_and_edge():
    from training.games.shogi_legal import legal_moves
    s = "9/9/9/9/4P4/9/9/9/4K4 b P 1"
    moves = legal_moves(s)
    dropp = [m for m in moves if m.startswith("D") and m.endswith("p")]
    files = {int(m[1:-1]) % 9 for m in dropp}
    assert 4 not in files
    assert not any(int(m[1:-1]) // 9 == 0 for m in dropp)


def test_pawn_drop_mate_excluded():
    from training.games.shogi_legal import legal_moves
    s = "4k4/9/9/9/9/9/9/9/4K4 b P 1"
    moves = legal_moves(s)
    assert "D76p" not in moves


def test_check_evasion():
    from training.games.shogi import parse_sfen
    from training.games.shogi_legal import apply_move, king_in_check, legal_moves
    s = "4k4/9/9/4R4/9/9/9/9/4K4 w - 1"
    board, stm, _, _ = parse_sfen(s)
    assert king_in_check(board, 1)
    moves = legal_moves(s)
    assert len(moves) > 0
    for mv in moves:
        nxt = apply_move(s, mv)
        b2, _, _, _ = parse_sfen(nxt)
        assert not king_in_check(b2, 1)


def test_mate_is_loss():
    from training.games.shogi_legal import game_result, legal_moves
    mate = "4k4/4G4/4G4/9/9/9/9/9/4K4 w - 1"
    assert legal_moves(mate) == []
    assert game_result(mate) == "1-0"
    assert game_result(START) == "*"
