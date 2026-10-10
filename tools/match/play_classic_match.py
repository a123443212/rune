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

import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def elo_from_score(score, games):
    if games == 0:
        return 0.0, 0.0
    s = min(max(score / games, 1e-6), 1 - 1e-6)
    elo = -400.0 * math.log10(1.0 / s - 1.0)
    se = math.sqrt(s * (1 - s) / games)
    err = 1.96 * se * 400.0 / (s * (1 - s) * math.log(10))
    return elo, err


def load_game(game_id):
    if game_id == "xiangqi":
        from training.games.xiangqi_legal import apply_move, game_result, legal_moves
        start = "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1"
        return legal_moves, apply_move, game_result, start
    from training.games.shogi_legal import apply_move, game_result, legal_moves
    start = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    return legal_moves, apply_move, game_result, start


def play_one(legal_fn, apply_fn, result_fn, start, eval_a, eval_b, sims, max_moves, seed, scorer=None, margin=0.0):
    from training.search.mcts import PuctSearch
    from training.search.token_adapter import uniform_priors
    rng = random.Random(seed)
    state = start
    moves = 0
    while moves < max_moves:
        legal = legal_fn(state)
        if not legal:
            break
        ev = eval_a if (moves % 2 == 0) else eval_b
        v0 = ev(state)
        search = PuctSearch(simulations=sims, seed=rng.randint(0, 10 ** 9))
        mv, _ = search.search(state, legal_fn, apply_fn, lambda s, _v=v0, _l=legal: (_v, uniform_priors(_l)))
        if mv is None:
            mv = legal[0]
        state = apply_fn(state, mv)
        moves += 1
        if result_fn(state) != "*":
            break
    res = result_fn(state)
    if res == "1-0":
        return 1.0
    if res == "0-1":
        return 0.0
    if scorer is not None and margin > 0.0:
        edge = scorer(state)
        if moves % 2 == 1:
            edge = -edge
        if edge > margin:
            return 1.0
        if edge < -margin:
            return 0.0
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=str, default="xiangqi")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--sims", type=int, default=8)
    ap.add_argument("--max-moves", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=70.0)
    ap.add_argument("--adjudicate-margin", type=float, default=0.0)
    args = ap.parse_args()
    if args.game not in ("xiangqi", "shogi"):
        raise ValueError("bad game")
    legal_fn, apply_fn, result_fn, start = load_game(args.game)
    from training.search.classic_eval import material_eval
    scorer = material_eval(args.game)

    def eval_a(state):
        return max(-1.0, min(1.0, scorer(state) / 2000.0))

    def eval_b(state):
        return -eval_a(state)

    wins = 0.0
    nw = 0
    nl = 0
    nd = 0
    t0 = time.perf_counter()
    for i in range(args.games):
        if i % 2 == 0:
            r = play_one(legal_fn, apply_fn, result_fn, start, eval_a, eval_b, args.sims, args.max_moves, args.seed + i, scorer, args.adjudicate_margin)
        else:
            r = play_one(legal_fn, apply_fn, result_fn, start, eval_b, eval_a, args.sims, args.max_moves, args.seed + i, scorer, args.adjudicate_margin)
            r = 1.0 - r
        wins += r
        if r == 1.0:
            nw += 1
        elif r == 0.0:
            nl += 1
        else:
            nd += 1
    dt = time.perf_counter() - t0
    elo, err = elo_from_score(wins, args.games)
    from tools.match.go_sprt import sprt_bounds, sprt_decide, sprt_llr
    llr = sprt_llr(nw, nl, nd, args.elo0, args.elo1)
    lower, upper = sprt_bounds()
    decision = sprt_decide(llr, lower, upper)
    print(f"game={args.game} games={args.games} score={wins:.1f} elo={elo:.1f}+-{err:.1f} seconds={dt:.2f}")
    print(f"w={nw} l={nl} d={nd} llr={llr:.3f} bounds=[{lower:.3f},{upper:.3f}] sprt={decision}")


if __name__ == "__main__":
    main()
