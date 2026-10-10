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

TEACHER_ID = "mcts_go_v02"


def distilled_label(game, model, state, simulations=32, seed=1):
    from training.search.go_adapter import search_move
    t0 = time.perf_counter()
    mv, info = search_move(game, model, state, simulations=simulations, seed=seed)
    dt = time.perf_counter() - t0
    legal = game.legal(state)
    n = int(game.size)
    policy_size = n * n + 1
    policy = [0.0] * policy_size
    if mv is None:
        mv = -1
    if int(mv) == -1:
        policy[n * n] = 1.0
    else:
        policy[int(mv)] = 1.0
    import torch
    planes = game.planes(state)
    with torch.no_grad():
        v, w, _, _ = model(torch.tensor(planes))
    return {
        "game": "go",
        "state": state,
        "move": int(mv),
        "value": float(v[0]),
        "wdl": [float(x) for x in w[0]],
        "policy": policy,
        "legal": [int(m) for m in legal],
        "teacher": TEACHER_ID,
        "sims": int(simulations),
        "seed": int(seed),
        "seconds": dt,
        "feature_version": game.feature_version,
    }


def distill_states(game, model, states, simulations=32, seed=1):
    out = []
    for i, s in enumerate(states):
        out.append(distilled_label(game, model, s, simulations=simulations, seed=seed + i))
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
        "storage_bytes": 648 * n,
    }
