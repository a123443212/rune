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

from training.games.shogi import HAND_TYPES, TYPES, on_board, slide_dirs, sq_file, sq_rank, step_moves


def promotable(pt):
    return pt in ("p", "l", "n", "s", "b", "r")


def unpromote(pt):
    return {"pp": "p", "pl": "l", "pn": "n", "ps": "s", "hb": "b", "dr": "r"}.get(pt, pt)


def in_zone(rank, color):
    if color == 0:
        return rank <= 2
    return rank >= 6


def must_promote(pt, color, to_rank):
    last = 0 if color == 0 else 8
    if pt in ("p", "l"):
        return to_rank == last
    if pt == "n":
        if color == 0:
            return to_rank <= 1
        return to_rank >= 7
    return False


def promo_options(board, frm, to, color):
    cell = board[frm]
    if cell is None:
        return []
    pt, _ = cell
    if not promotable(pt):
        return [False]
    if in_zone(sq_rank(frm), color) or in_zone(sq_rank(to), color):
        if must_promote(pt, color, sq_rank(to)):
            return [True]
        return [False, True]
    return [False]


def step_dests(board, frm, color):
    cell = board[frm]
    pt, _ = cell
    out = []
    for df, dr in step_moves(pt, color):
        tf = sq_file(frm) + df
        tr = sq_rank(frm) + dr
        if on_board(tf, tr):
            out.append(tr * 9 + tf)
    if pt == "l":
        f = -1 if color == 0 else 1
        tf = sq_file(frm)
        tr = sq_rank(frm) + f
        while on_board(tf, tr):
            t = tr * 9 + tf
            out.append(t)
            if board[t] is not None:
                break
            tr += f
    dirs = slide_dirs(pt)
    if dirs is None:
        dirs = []
    for df, dr in dirs:
        tf = sq_file(frm) + df
        tr = sq_rank(frm) + dr
        while on_board(tf, tr):
            t = tr * 9 + tf
            out.append(t)
            if board[t] is not None:
                break
            tf += df
            tr += dr
    return out


def drop_squares(board, hand_count, piece, color):
    out = []
    for sq in range(81):
        if board[sq] is not None:
            continue
        r = sq_rank(sq)
        f = sq_file(sq)
        if piece == "p" or piece == "l":
            if (color == 0 and r == 0) or (color == 1 and r == 8):
                continue
        if piece == "n":
            if (color == 0 and r <= 1) or (color == 1 and r >= 7):
                continue
        if piece == "p":
            bad = False
            for rr in range(9):
                c = board[rr * 9 + f]
                if c is not None and c == ("p", color):
                    bad = True
                    break
            if bad:
                continue
        if hand_count <= 0:
            continue
        out.append(sq)
    return out


def pseudo_moves(board, hand, stm):
    out = []
    for frm in range(81):
        cell = board[frm]
        if cell is None or cell[1] != stm:
            continue
        for to in step_dests(board, frm, stm):
            target = board[to]
            if target is not None and target[1] == stm:
                continue
            for promo in promo_options(board, frm, to, stm):
                out.append((frm, to, promo))
    for t in HAND_TYPES:
        n = hand.get((t, stm), 0)
        for to in drop_squares(board, n, t, stm):
            out.append((None, to, t))
    return out
