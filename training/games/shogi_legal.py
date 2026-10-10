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

from training.games.shogi import HAND_TYPES, TYPES, parse_sfen, piece_attacks
from training.games.shogi_moves import pseudo_moves, unpromote


def encode_move(fr, to, promo):
    if fr is None:
        return "D" + str(to) + str(promo)
    s = f"{fr:02d}{to:02d}"
    if promo is True:
        return s + "+"
    return s


def decode_move(mv):
    if mv.startswith("D"):
        body = mv[1:]
        for t in HAND_TYPES:
            if body.endswith(t):
                try:
                    to = int(body[: -len(t)])
                except ValueError:
                    raise ValueError("bad shogi move")
                if to < 0 or to >= 81:
                    raise ValueError("bad shogi move")
                return None, to, t
        raise ValueError("bad shogi move")
    promo = mv.endswith("+")
    core = mv[:-1] if promo else mv
    if len(core) != 4:
        raise ValueError("bad shogi move")
    try:
        fr = int(core[:2])
        to = int(core[2:])
    except ValueError:
        raise ValueError("bad shogi move")
    if fr < 0 or fr >= 81 or to < 0 or to >= 81:
        raise ValueError("bad shogi move")
    return fr, to, promo


def king_in_check(board, color):
    kus = -1
    for sq in range(81):
        if board[sq] == ("k", color):
            kus = sq
            break
    if kus < 0:
        return True
    for asq in range(81):
        attacker = board[asq]
        if attacker is None or attacker[1] == color:
            continue
        if piece_attacks(board, asq, kus):
            return True
    return False


def do_move(board, hand, stm, fr, to, promo):
    nb = list(board)
    nh = dict(hand)
    if fr is None:
        if nh.get((promo, stm), 0) <= 0:
            raise ValueError("bad shogi move")
        if nb[to] is not None:
            raise ValueError("bad shogi move")
        nh[(promo, stm)] = nh[(promo, stm)] - 1
        nb[to] = (promo, stm)
        return nb, nh
    cell = nb[fr]
    if cell is None or cell[1] != stm:
        raise ValueError("bad shogi move")
    pt, _ = cell
    if promo is True:
        promo_map = {"p": "pp", "l": "pl", "n": "pn", "s": "ps", "b": "hb", "r": "dr"}
        if pt not in promo_map:
            raise ValueError("bad shogi move")
        pt = promo_map[pt]
    target = nb[to]
    if target is not None and target[1] == stm:
        raise ValueError("bad shogi move")
    if target is not None:
        base = unpromote(target[0])
        nh[(base, stm)] = nh.get((base, stm), 0) + 1
    nb[to] = (pt, stm)
    nb[fr] = None
    return nb, nh


def serialize_sfen(board, stm, hand, move_no):
    ranks = []
    for ri in range(9):
        rank = ""
        gap = 0
        for f in range(9):
            cell = board[ri * 9 + f]
            if cell is None:
                gap += 1
                continue
            if gap:
                rank += str(gap)
                gap = 0
            pt, color = cell
            if len(pt) > 1:
                base = unpromote(pt)
                rank += "+" + (base.upper() if color == 0 else base)
            else:
                rank += pt.upper() if color == 0 else pt
        if gap:
            rank += str(gap)
        ranks.append(rank)
    grid = "/".join(ranks)
    side = "b" if stm == 0 else "w"
    parts = []
    for color in (0, 1):
        for t in ("r", "b", "g", "s", "n", "l", "p"):
            n = hand.get((t, color), 0)
            if n <= 0:
                continue
            if n > 1:
                parts.append(str(n))
            parts.append(t.upper() if color == 0 else t)
    hands = "".join(parts) if parts else "-"
    return f"{grid} {side} {hands} {move_no}"


def pawn_drop_mate(board, hand, stm, to):
    nb = list(board)
    nb[to] = ("p", stm)
    if not king_in_check(nb, 1 - stm):
        return False
    foe = 1 - stm
    for frm in range(81):
        cell = nb[frm]
        if cell is None or cell[1] != foe:
            continue
        from training.games.shogi_moves import step_dests
        for t in step_dests(nb, frm, foe):
            target = nb[t]
            if target is not None and target[1] == foe:
                continue
            if t == to:
                return False
            trial = list(nb)
            trial[t] = trial[frm]
            trial[frm] = None
            if not king_in_check(trial, foe):
                return False
    return True


def legal_moves(state):
    board, stm, hand, _ = parse_sfen(state)
    out = []
    for fr, to, promo in pseudo_moves(board, hand, stm):
        if fr is None:
            if board[to] is not None:
                continue
            if promo == "p" and pawn_drop_mate(board, hand, stm, to):
                continue
            nb = list(board)
            nb[to] = (promo, stm)
            if king_in_check(nb, stm):
                continue
            out.append(encode_move(fr, to, promo))
            continue
        target = board[to]
        if target is not None and target[1] == stm:
            continue
        nb, _ = do_move(board, hand, stm, fr, to, promo)
        if king_in_check(nb, stm):
            continue
        out.append(encode_move(fr, to, promo))
    return sorted(out)


def apply_move(state, mv):
    fr, to, promo = decode_move(mv)
    board, stm, hand, move_no = parse_sfen(state)
    if fr is None:
        if board[to] is not None:
            raise ValueError("bad shogi move")
        if promo == "p" and pawn_drop_mate(board, hand, stm, to):
            raise ValueError("bad shogi move")
    nb, nh = do_move(board, hand, stm, fr, to, promo)
    if king_in_check(nb, stm):
        raise ValueError("bad shogi move")
    return serialize_sfen(nb, 1 - stm, nh, move_no + 1)


def game_result(state):
    _, stm, _, _ = parse_sfen(state)
    if not legal_moves(state):
        if stm == 0:
            return "0-1"
        return "1-0"
    return "*"
