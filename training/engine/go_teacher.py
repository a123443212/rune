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

import math

TEACHER_ID = "synthetic_go_v02"
LABEL_SECONDS_PER_POS = 0.0002
STORAGE_BYTES_PER_POS = 648


def value_from_score(score):
    v = math.tanh(float(score) / 12.0)
    if v > 1.0:
        return 1.0
    if v < -1.0:
        return -1.0
    return v


def wdl_from_value(value):
    p = 1.0 / (1.0 + math.exp(-3.0 * float(value)))
    d = 0.05
    w = p * (1.0 - d)
    l = (1.0 - p) * (1.0 - d)
    return [w, d, l]


def label_state(game, state):
    score = float(game.score(state))
    value = value_from_score(score)
    wdl = wdl_from_value(value)
    legal = game.legal(state)
    k = float(len(legal))
    policy = [1.0 / k for _ in legal]
    return {
        "game": "go",
        "state": state,
        "value": value,
        "wdl": wdl,
        "policy": policy,
        "legal": [int(m) for m in legal],
        "score": score,
        "teacher": TEACHER_ID,
        "feature_version": game.feature_version,
    }


def label_states(game, states):
    return [label_state(game, s) for s in states]


def teacher_cost(n):
    n = int(n)
    return {
        "teacher": TEACHER_ID,
        "positions": n,
        "generation_seconds": 0.0,
        "labeling_seconds": LABEL_SECONDS_PER_POS * n,
        "storage_bytes": STORAGE_BYTES_PER_POS * n,
    }
