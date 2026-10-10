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

from training.games.base import GameSpec

TYPES = ["p", "h", "r", "c", "a", "e", "k"]
MAJORS = ["h", "r", "c"]

FEATURE_VERSION = "xiangqi_raw_v01"
VOCABS = [198, 180, 360, 180, 360, 639, 694, 64, 18]
CONTEXT_DIM = 12
CONTEXT_NAMES = ["stm", "phase", "us_majors", "them_majors", "us_pawns",
                 "them_pawns", "us_crossed", "check", "flying",
                 "us_palace", "mat_lead", "move_no"]


def sq_file(sq):
    return sq % 9


def sq_rank(sq):
    return sq // 9


def make_sq(f, r):
    return r * 9 + f


def on_board(f, r):
    return 0 <= f < 9 and 0 <= r < 10


def in_palace(f, r, color):
    if f < 3 or f > 5:
        return False
    if color == 0:
        return 0 <= r <= 2
    return 7 <= r <= 9


def crossed(r, color):
    if color == 0:
        return r >= 5
    return r <= 4


def parse_fen(fen):
    parts = fen.split()
    placement = parts[0]
    side = parts[1] if len(parts) > 1 else "w"
    board = [None] * 90
    ranks = placement.split("/")
    if len(ranks) != 10:
        raise ValueError(f"bad xiangqi fen board {fen}")
    for ri, rank in enumerate(ranks):
        r = 9 - ri
        f = 0
        for c in rank:
            if c.isdigit():
                f += int(c)
            else:
                if c.lower() not in TYPES:
                    raise ValueError(f"bad xiangqi piece {fen}")
                if f >= 9:
                    raise ValueError(f"bad xiangqi rank {fen}")
                color = 0 if c.isupper() else 1
                board[make_sq(f, r)] = (c.lower(), color)
                f += 1
        if f != 9:
            raise ValueError(f"bad xiangqi rank width {fen}")
    stm = 0 if side in ("w", "r") else 1
    move_no = 1
    if len(parts) > 5:
        try:
            move_no = max(1, int(parts[5]))
        except ValueError:
            move_no = 1
    return board, stm, move_no


def sliding_clear(board, frm, target, df, dr):
    f, r = sq_file(frm) + df, sq_rank(frm) + dr
    while (f, r) != (sq_file(target), sq_rank(target)):
        if board[make_sq(f, r)] is not None:
            return False
        f += df
        r += dr
    return True


def piece_attacks(board, frm, target):
    if board[frm] is None or frm == target:
        return False
    p, color = board[frm]
    ff, rf = sq_file(frm), sq_rank(frm)
    tf, rt = sq_file(target), sq_rank(target)
    df, dr = tf - ff, rt - rf
    adf, adr = abs(df), abs(dr)
    fwd = 1 if color == 0 else -1
    if p == "k":
        if adf + adr == 1 and in_palace(tf, rt, color):
            return True
        if board[target] is not None and board[target][0] == "k":
            if tf == ff:
                lo, hi = (frm, target) if frm < target else (target, frm)
                for sq in range(lo + 9, hi, 9):
                    if board[sq] is not None:
                        return False
                return True
        return False
    if p == "a":
        if not in_palace(tf, rt, color):
            return False
        return adf == 1 and adr == 1
    if p == "e":
        if adf != 2 or adr != 2:
            return False
        if color == 0 and rt > 4:
            return False
        if color == 1 and rt < 5:
            return False
        return board[make_sq(ff + df // 2, rf + dr // 2)] is None
    if p == "h":
        if (adf, adr) not in ((1, 2), (2, 1)):
            return False
        if adf == 2:
            leg = make_sq(ff + df // 2, rf)
        else:
            leg = make_sq(ff, rf + dr // 2)
        if board[leg] is not None:
            return False
        return True
    if p == "r":
        if df != 0 and dr != 0:
            return False
        sf = 0 if df == 0 else (1 if df > 0 else -1)
        sr = 0 if dr == 0 else (1 if dr > 0 else -1)
        return sliding_clear(board, frm, target, sf, sr)
    if p == "c":
        if df != 0 and dr != 0:
            return False
        sf = 0 if df == 0 else (1 if df > 0 else -1)
        sr = 0 if dr == 0 else (1 if dr > 0 else -1)
        f, r = ff + sf, rf + sr
        screens = 0
        while (f, r) != (tf, rt):
            if board[make_sq(f, r)] is not None:
                screens += 1
            f += sf
            r += sr
        return screens == 1
    if p == "p":
        if df == 0 and dr == fwd:
            return True
        if dr == 0 and adf == 1 and crossed(rf, color):
            return True
        return False
    return False


def flying(board):
    kings = {}
    for sq in range(90):
        if board[sq] is not None and board[sq][0] == "k":
            kings[board[sq][1]] = sq
    if 0 not in kings or 1 not in kings:
        return False
    a, b = kings[0], kings[1]
    if sq_file(a) != sq_file(b):
        return False
    lo, hi = (a, b) if a < b else (b, a)
    for sq in range(lo + 9, hi, 9):
        if board[sq] is not None:
            return False
    return True


def clamp01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def game_phase(board):
    n = 0
    for cell in board:
        if cell is None:
            continue
        if cell[0] in ("a", "e", "h", "r", "c"):
            n += 1
    if n >= 14:
        return 0
    if n >= 8:
        return 1
    return 2


class XiangqiGame(GameSpec):
    game_id = "xiangqi"
    num_groups = 9
    vocabs = tuple(VOCABS)
    tokens = 8
    token_dim = 32
    context_dim = CONTEXT_DIM
    context_names = tuple(CONTEXT_NAMES)
    feature_version = FEATURE_VERSION

    def extract(self, state):
        board, stm, move_no = parse_fen(state)
        us = stm

        def rel(sq):
            return sq

        feats = []
        for sq in range(90):
            cell = board[sq]
            if cell is None:
                continue
            p, color = cell
            ci = 0 if color == us else 1
            s = rel(sq)
            if p == "p":
                feats.append((0, ci * 90 + s))
                if crossed(sq_rank(sq), color):
                    feats.append((0, 180 + ci * 9 + sq_file(sq)))
            if p == "k":
                feats.append((1, ci * 90 + s))
            if p in ("a", "e"):
                feats.append((2, (ci * 2 + TYPES.index(p) - 4) * 90 + s))
            if p == "h":
                feats.append((3, ci * 90 + s))
            if p in ("r", "c"):
                feats.append((4, (ci * 2 + TYPES.index(p) - 2) * 90 + s))
            if p in ("p", "h", "r", "c", "a", "e", "k"):
                feats.append((6, TYPES.index(p) * 90 + s))
        fly = flying(board)
        for vsq in range(90):
            victim = board[vsq]
            if victim is None:
                continue
            for asq in range(90):
                attacker = board[asq]
                if attacker is None or attacker[1] == victim[1]:
                    continue
                if not piece_attacks(board, asq, vsq):
                    continue
                vt = TYPES.index(victim[0])
                feats.append((5, vt * 90 + rel(vsq)))
                feats.append((5, 630 + sq_file(rel(asq))))
        for f in range(9):
            majors = 0
            for r in range(10):
                cell = board[make_sq(f, r)]
                if cell is not None and cell[0] in MAJORS and cell[1] == us:
                    majors += 1
            if majors >= 2:
                feats.append((8, f))
            majors_t = 0
            for r in range(10):
                cell = board[make_sq(f, r)]
                if cell is not None and cell[0] in MAJORS and cell[1] != us:
                    majors_t += 1
            if majors_t >= 2:
                feats.append((8, 9 + f))
        feats.append((7, stm))
        feats.append((7, 2 + game_phase(board)))
        total = sum(1 for c in board if c is not None)
        feats.append((7, 5 + min((32 - total) // 2, 15)))
        kus = -1
        for sq in range(90):
            if board[sq] == ("k", us):
                kus = sq
                break
        check = 0
        if kus >= 0:
            for asq in range(90):
                attacker = board[asq]
                if attacker is None or attacker[1] == us:
                    continue
                if piece_attacks(board, asq, kus):
                    check = 1
                    break
            if not check and fly:
                check = 1
        if check:
            feats.append((7, 21))
        if fly:
            feats.append((7, 22))
        feats.append((7, 23 + min(move_no // 20, 5)))
        return sorted(set(feats))

    def context(self, state):
        board, stm, move_no = parse_fen(state)
        us = stm
        us_majors = 0
        them_majors = 0
        us_pawns = 0
        them_pawns = 0
        us_crossed = 0
        us_palace = 0
        us_total = 0
        them_total = 0
        for sq in range(90):
            cell = board[sq]
            if cell is None:
                continue
            p, color = cell
            if color == us:
                us_total += 1
                if p in MAJORS:
                    us_majors += 1
                if p == "p":
                    us_pawns += 1
                    if crossed(sq_rank(sq), color):
                        us_crossed += 1
                if p in ("a", "e"):
                    us_palace += 1
            else:
                them_total += 1
                if p in MAJORS:
                    them_majors += 1
                if p == "p":
                    them_pawns += 1
        kus = -1
        for sq in range(90):
            if board[sq] == ("k", us):
                kus = sq
                break
        check = 0
        if kus >= 0:
            for asq in range(90):
                attacker = board[asq]
                if attacker is None or attacker[1] == us:
                    continue
                if piece_attacks(board, asq, kus):
                    check = 1
                    break
            if not check and flying(board):
                check = 1
        fly = 1 if flying(board) else 0
        lead = (us_total - them_total + 16) / 32.0
        return [clamp01(float(stm)), clamp01(game_phase(board) / 2.0),
                clamp01(us_majors / 12.0), clamp01(them_majors / 12.0),
                clamp01(us_pawns / 5.0), clamp01(them_pawns / 5.0),
                clamp01(us_crossed / 5.0), clamp01(float(check)),
                clamp01(float(fly)), clamp01(us_palace / 4.0),
                clamp01(lead), clamp01(move_no / 200.0)]

    def phase(self, state):
        board, _, _ = parse_fen(state)
        return game_phase(board)

    def normalize(self, state):
        parts = state.split()
        return " ".join(parts[:2])

    def phase_from_ids(self, group_ids, group_mask):
        g7 = group_ids[7].clamp(0, 63)
        valid = (group_mask[7] > 0.5) & (g7 >= 2) & (g7 <= 4)
        cand = torch.where(valid, g7, torch.full_like(g7, 2))
        return (cand.amax(dim=1) - 2).clamp(0, 2).long()
