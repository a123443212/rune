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

from training.games.xiangqi import parse_fen, piece_attacks
from training.games.xiangqi_moves import pseudo_dests


def encode_move(fr, to):
    return f"{fr:02d}{to:02d}"


def decode_move(mv):
    if len(mv) != 4:
        raise ValueError("bad xiangqi move")
    try:
        fr = int(mv[:2])
        to = int(mv[2:])
    except ValueError:
        raise ValueError("bad xiangqi move")
    if fr < 0 or fr >= 90 or to < 0 or to >= 90:
        raise ValueError("bad xiangqi move")
    return fr, to


def king_in_check(board, color):
    kus = -1
    for sq in range(90):
        if board[sq] == ("k", color):
            kus = sq
            break
    if kus < 0:
        return True
    for asq in range(90):
        attacker = board[asq]
        if attacker is None or attacker[1] == color:
            continue
        if piece_attacks(board, asq, kus):
            return True
    return False


def apply_on_board(board, stm, fr, to):
    if board[fr] is None or board[fr][1] != stm:
        raise ValueError("bad xiangqi move")
    nb = list(board)
    nb[to] = nb[fr]
    nb[fr] = None
    if king_in_check(nb, stm):
        raise ValueError("bad xiangqi move")
    return nb


def legal_moves(state):
    board, stm, _ = parse_fen(state)
    out = []
    for fr in range(90):
        cell = board[fr]
        if cell is None or cell[1] != stm:
            continue
        for to in pseudo_dests(board, fr):
            target = board[to]
            if target is not None and target[1] == stm:
                continue
            nb = list(board)
            nb[to] = nb[fr]
            nb[fr] = None
            if king_in_check(nb, stm):
                continue
            out.append(encode_move(fr, to))
    return sorted(out)


def apply_move(state, mv):
    fr, to = decode_move(mv)
    board, stm, move_no = parse_fen(state)
    if board[fr] is None or board[fr][1] != stm:
        raise ValueError("bad xiangqi move")
    if to not in pseudo_dests(board, fr):
        raise ValueError("bad xiangqi move")
    target = board[to]
    if target is not None and target[1] == stm:
        raise ValueError("bad xiangqi move")
    nb = apply_on_board(board, stm, fr, to)
    rows = []
    for r in range(9, -1, -1):
        rank = ""
        gap = 0
        for f in range(9):
            cell = nb[r * 9 + f]
            if cell is None:
                gap += 1
                continue
            if gap:
                rank += str(gap)
                gap = 0
            p, color = cell
            rank += p.upper() if color == 0 else p
        if gap:
            rank += str(gap)
        rows.append(rank)
    grid = "/".join(rows)
    nstm = 1 - stm
    side = "w" if nstm == 0 else "b"
    nmove = move_no + (1 if stm == 1 else 0)
    return f"{grid} {side} - - 0 {nmove}"


def game_result(state):
    _, stm, _ = parse_fen(state)
    if not legal_moves(state):
        if stm == 0:
            return "0-1"
        return "1-0"
    return "*"
