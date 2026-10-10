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
import json
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


def try_bindings():
    try:
        import rune_bindings as rb
        return rb
    except Exception:
        return None


def eval_for(rb_board, model, stats):
    t0 = time.perf_counter()
    try:
        out = model.eval_fen(rb_board.to_fen())
    except TypeError:
        out = model.eval_fen(rb_board.to_fen(), "full", 0.0)
    dt = time.perf_counter() - t0
    stats["evals"] += 1
    stats["seconds"] += dt
    if isinstance(out, (list, tuple)):
        return float(out[0])
    return float(out)


def negamax(rb, board, model, stats, depth, alpha, beta):
    stats["nodes"] += 1
    if depth <= 0:
        return eval_for(board, model, stats)
    moves = board.legal_moves()
    if not moves:
        return eval_for(board, model, stats)
    best = -1e9
    for m in moves:
        mv = rb.Move()
        mv.from_sq, mv.to_sq, mv.promo = m[0], m[1], m[2]
        if not board.make_move(mv):
            continue
        v = -negamax(rb, board, model, stats, depth - 1, -beta, -alpha)
        board.unmake_move()
        if v > best:
            best = v
        if v > alpha:
            alpha = v
        if alpha >= beta:
            stats["cutoffs"] += 1
            break
    return best


def best_move(rb, board, model, stats, depth):
    moves = board.legal_moves()
    if not moves:
        return None
    scored = []
    for m in moves:
        mv = rb.Move()
        mv.from_sq, mv.to_sq, mv.promo = m[0], m[1], m[2]
        if not board.make_move(mv):
            continue
        v = -negamax(rb, board, model, stats, depth - 1, -1e9, 1e9)
        board.unmake_move()
        scored.append((-v, m))
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def play_game(rb, white_model, black_model, ws, bs, opening_fen, depth, max_ply=120):
    b = rb.Board(opening_fen)
    for _ in range(max_ply):
        moves = b.legal_moves()
        if not moves:
            return 0.5
        stm = b.side_to_move()
        model = white_model if stm == 0 else black_model
        stats = ws if stm == 0 else bs
        m = best_move(rb, b, model, stats, depth)
        if m is None:
            return 0.5
        mv = rb.Move()
        mv.from_sq, mv.to_sq, mv.promo = m[0], m[1], m[2]
        b.make_move(mv)
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--report", default="")
    args = ap.parse_args()
    rb = try_bindings()
    if rb is None:
        print("rune_bindings unavailable, match skipped")
        return 3
    sys.path.insert(0, args.build_dir)
    import rune_bindings as rb2
    rb = rb2
    from tools.match.play_match import load_model_for_match
    ma = load_model_for_match(None, args.model_a, args.build_dir)
    mb = load_model_for_match(None, args.model_b, args.build_dir)
    rng = random.Random(args.seed)
    openings = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"]
    sa = {"nodes": 0, "evals": 0, "seconds": 0.0, "cutoffs": 0}
    sb = {"nodes": 0, "evals": 0, "seconds": 0.0, "cutoffs": 0}
    score = 0.0
    for i in range(args.games):
        if i % 2 == 0:
            score += play_game(rb, ma, mb, sa, sb, openings[i % len(openings)], args.depth)
        else:
            score += 1.0 - play_game(rb, mb, ma, sb, sa, openings[i % len(openings)], args.depth)
    elo, err = elo_from_score(score, args.games)
    print("score A: %s/%s elo: %.1f +/- %.1f" % (score, args.games, elo, err))
    print("A nodes %d evals %d nps %.0f" % (sa["nodes"], sa["evals"], sa["nodes"] / max(1e-9, sa["seconds"])))
    print("B nodes %d evals %d nps %.0f" % (sb["nodes"], sb["evals"], sb["nodes"] / max(1e-9, sb["seconds"])))
    if args.report:
        json.dump({"score_a": score, "games": args.games, "elo_a": elo, "elo_err95": err, "depth": args.depth, "seed": args.seed, "a": sa, "b": sb}, open(args.report, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
