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
from training.games.xiangqi import XiangqiGame, parse_fen, piece_attacks
from training.games.xiangqi_moves import king_dests

FEATURE_VERSION_V02 = "xiangqi_sem_v02"


def double_attacks(board, stm):
    out = []
    for vsq in range(90):
        victim = board[vsq]
        if victim is None or victim[1] == stm:
            continue
        n = 0
        for asq in range(90):
            attacker = board[asq]
            if attacker is None or attacker[1] != stm:
                continue
            if piece_attacks(board, asq, vsq):
                n += 1
                if n >= 2:
                    break
        if n >= 2:
            out.append(vsq)
    return sorted(out)


def king_mobility(board, stm):
    kus = -1
    for sq in range(90):
        if board[sq] == ("k", stm):
            kus = sq
            break
    if kus < 0:
        return 0
    n = 0
    for to in king_dests(board, kus, stm):
        target = board[to]
        if target is not None and target[1] == stm:
            continue
        hit = False
        for asq in range(90):
            attacker = board[asq]
            if attacker is None or attacker[1] == stm:
                continue
            if piece_attacks(board, asq, to):
                hit = True
                break
        if not hit:
            n += 1
    return n


class XiangqiGameV02(GameSpec):
    game_id = "xiangqi"
    num_groups = 9
    vocabs = (198, 180, 360, 180, 360, 639, 694, 64, 18)
    tokens = 8
    token_dim = 32
    context_dim = 12
    context_names = ("stm", "phase", "us_majors", "them_majors", "us_pawns", "them_pawns",
                     "us_crossed", "check", "flying", "us_palace", "mat_lead", "move_no")
    feature_version = FEATURE_VERSION_V02

    def __init__(self):
        self.v01 = XiangqiGame()

    def extract(self, state):
        feats = self.v01.extract(state)
        board, stm, _ = parse_fen(state)
        doubles = double_attacks(board, stm)
        mob = king_mobility(board, stm)
        out = list(feats)
        for sq in doubles:
            out.append((6, 630 + (sq % 64)))
        out.append((7, 29 + min(len(doubles), 9)))
        out.append((7, 39 + min(mob, 4)))
        return sorted(set(out))

    def context(self, state):
        return self.v01.context(state)

    def phase(self, state):
        return self.v01.phase(state)

    def normalize(self, state):
        return self.v01.normalize(state)

    def phase_from_ids(self, group_ids, group_mask):
        g7 = group_ids[7].clamp(0, 63)
        valid = (group_mask[7] > 0.5) & (g7 >= 2) & (g7 <= 4)
        cand = torch.where(valid, g7, torch.full_like(g7, 2))
        return (cand.amax(dim=1) - 2).clamp(0, 2).long()
