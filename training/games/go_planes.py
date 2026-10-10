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

import numpy as np

from training.games.go_board import liberty_map
from training.games.go_state import decode_state, komi_bucket, move_bucket


PLANES_V02 = 8


def planes_v02(state, size=9):
    d = decode_state(state, size)
    board = d["board"]
    n = d["n"]
    stm = d["stm"]
    ko = d["ko"]
    libs = liberty_map(board, n)
    out = np.zeros((1, PLANES_V02, n, n), dtype=np.float32)
    for r in range(n):
        for c in range(n):
            sq = r * n + c
            v = board[sq]
            mine = (v == 1 and stm == 0) or (v == -1 and stm == 1)
            theirs = (v == 1 or v == -1) and not mine and v != 0
            if mine:
                out[0, 0, r, c] = 1.0
            if theirs:
                out[0, 1, r, c] = 1.0
            if v == 0:
                out[0, 2, r, c] = 1.0
            else:
                k = libs[sq]
                if k <= 1:
                    out[0, 3, r, c] = 1.0
                elif k == 2:
                    out[0, 4, r, c] = 1.0
                else:
                    out[0, 5, r, c] = 1.0
            if ko is not None and sq == ko:
                out[0, 6, r, c] = 1.0
            out[0, 7, r, c] = float(stm)
    return out


def token_features_v02(state, size=9):
    d = decode_state(state, size)
    board = d["board"]
    n = d["n"]
    stm = d["stm"]
    ko = d["ko"]
    komi = d["komi"]
    move_no = d["move_no"]
    pass_no = d["pass_no"]
    libs = liberty_map(board, n)
    feats = []
    atari = 0
    denom = n * n
    if denom > 361:
        denom = 361
    for sq in range(n * n):
        v = board[sq]
        if v == 0:
            continue
        mine = (v == 1 and stm == 0) or (v == -1 and stm == 1)
        if mine:
            ci = 0
        else:
            ci = 1
        key = (ci * 181 + sq) % 361
        feats.append((0, key))
        feats.append((6, sq % 361))
        k = libs[sq]
        if k <= 2:
            feats.append((1, key))
        if k == 1:
            feats.append((5, sq % 361))
            atari += 1
    if ko is not None:
        feats.append((2, int(ko) % 361))
        feats.append((5, 360 - (int(ko) % 361)))
    feats.append((7, int(stm)))
    feats.append((7, 2 + komi_bucket(komi)))
    feats.append((7, 10 + move_bucket(move_no)))
    feats.append((7, 16 + min(int(pass_no), 2)))
    if ko is not None:
        feats.append((7, 19))
    mb = move_bucket(move_no)
    if mb <= 1:
        ph = 0
    elif mb <= 3:
        ph = 1
    else:
        ph = 2
    feats.append((7, 20 + ph))
    a = min(int(atari), 8)
    feats.append((8, a))
    if ko is not None:
        feats.append((8, 9 + (int(ko) % 9)))
    else:
        feats.append((8, 9))
    return sorted(set(feats))
