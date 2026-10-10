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

import time

TEACHER_ID = "search_classic_v01"


def distilled_label(game, model, state, legal_fn, apply_fn, simulations=32, seed=1):
    from training.search.token_adapter import search_move
    t0 = time.perf_counter()
    mv, info = search_move(game, model, state, legal_fn, apply_fn, simulations=simulations, seed=seed)
    dt = time.perf_counter() - t0
    legal = legal_fn(state)
    if mv is None:
        mv = legal[0] if legal else ""
    return {
        "game": game.game_id,
        "state": state,
        "move": mv,
        "legal": list(legal),
        "teacher": TEACHER_ID,
        "sims": int(simulations),
        "seed": int(seed),
        "seconds": dt,
        "root_visits": int(info.get("root_visits", 0)),
        "feature_version": game.feature_version,
    }


def distill_states(game, model, states, legal_fn, apply_fn, simulations=32, seed=1):
    out = []
    for i, s in enumerate(states):
        out.append(distilled_label(game, model, s, legal_fn, apply_fn, simulations=simulations, seed=seed + i))
    return out


def teacher_cost(records):
    n = len(records)
    secs = sum(float(r.get("seconds", 0.0)) for r in records)
    sims = sum(int(r.get("sims", 0)) for r in records)
    return {
        "teacher": TEACHER_ID,
        "positions": n,
        "simulations": sims,
        "labeling_seconds": secs,
        "storage_bytes": 256 * n,
    }
