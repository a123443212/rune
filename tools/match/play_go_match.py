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

import torch


def elo_from_score(score, games):
    if games == 0:
        return 0.0, 0.0
    s = min(max(score / games, 1e-6), 1 - 1e-6)
    elo = -400.0 * math.log10(1.0 / s - 1.0)
    se = math.sqrt(s * (1 - s) / games)
    err = 1.96 * se * 400.0 / (s * (1 - s) * math.log(10))
    return elo, err


def empty_state(n):
    return "/".join(["." * n] * n) + " b - 7.5 1 0"


def load_model(path, board, channels, blocks):
    from training.models.resnet import build_resnet
    from training.export.export import load_exported_arrays
    header, arrays = load_exported_arrays(path)
    in_planes = int(header.get("in_planes", 1))
    m = build_resnet(board=board, channels=channels, blocks=blocks, in_planes=in_planes, feature_set=header.get("feature_set", "go_planes_v02"))
    state_dict = m.state_dict()
    name_map = {"stem_w": "stem.weight", "stem_b": "stem.bias", "vh1": "value.fc1.weight", "bh1": "value.fc1.bias", "wv": "value.fcv.weight", "bv": "value.fcv.bias", "wwdl": "value.fcwdl.weight", "bwdl": "value.fcwdl.bias", "wpol": "policy.fc.weight", "bpol": "policy.fc.bias"}
    for i in range(blocks):
        name_map[f"b{i}_w1"] = f"tower.{i}.c1.weight"
        name_map[f"b{i}_b1"] = f"tower.{i}.c1.bias"
        name_map[f"b{i}_w2"] = f"tower.{i}.c2.weight"
        name_map[f"b{i}_b2"] = f"tower.{i}.c2.bias"
    for k, v in arrays.items():
        if k in name_map:
            target = state_dict[name_map[k]]
            state_dict[name_map[k]] = torch.tensor(v).reshape(target.shape)
    m.load_state_dict(state_dict)
    m.eval()
    return m


def synthetic_eval(game):
    def fn(state):
        legal = game.legal(state)
        k = float(len(legal))
        return 0.0, [1.0 / k for _ in legal]
    return fn


def model_eval(game, model):
    def fn(state):
        from training.search.go_adapter import build_go_callbacks
        _, _, eval_fn = build_go_callbacks(game, model)
        return eval_fn(state)
    return fn


def play_one(game, black_eval, white_eval, black_search, white_search, max_moves, seed):
    from training.search.mcts import PuctSearch
    state = empty_state(game.size)
    passes = 0
    moves = 0
    rng = random.Random(seed)
    while moves < max_moves:
        stm_is_black = state.split()[1] in ("b", "X")
        if stm_is_black:
            search = PuctSearch(simulations=black_search, seed=rng.randint(0, 10 ** 9))
            legal = game.legal(state)
            v0, p0 = black_eval(state)
            mv, _ = search.search(state, game.legal, game.apply, lambda s, _v=v0, _p=p0: (_v, _p))
        else:
            search = PuctSearch(simulations=white_search, seed=rng.randint(0, 10 ** 9))
            legal = game.legal(state)
            v0, p0 = white_eval(state)
            mv, _ = search.search(state, game.legal, game.apply, lambda s, _v=v0, _p=p0: (_v, _p))
        if mv is None:
            mv = -1
        state = game.apply(state, int(mv))
        moves += 1
        if int(mv) == -1:
            passes += 1
            if passes >= 2:
                break
        else:
            passes = 0
    score = game.score(state)
    if score > 0:
        return 1.0, state, moves
    if score < 0:
        return 0.0, state, moves
    return 0.5, state, moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--sims", type=int, default=8)
    ap.add_argument("--board", type=int, default=9)
    ap.add_argument("--max-moves", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model-a", type=str, default="")
    ap.add_argument("--model-b", type=str, default="")
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=70.0)
    args = ap.parse_args()
    from training.games.go_v02 import GoGameV02
    game = GoGameV02(size=args.board)
    if args.model_a:
        ma = load_model(args.model_a, args.board, args.channels, args.blocks)
        eval_a = model_eval(game, ma)
    else:
        eval_a = synthetic_eval(game)
    if args.model_b:
        mb = load_model(args.model_b, args.board, args.channels, args.blocks)
        eval_b = model_eval(game, mb)
    else:
        eval_b = synthetic_eval(game)
    wins = 0.0
    nw = 0
    nl = 0
    nd = 0
    t0 = time.perf_counter()
    for i in range(args.games):
        if i % 2 == 0:
            r, _, _ = play_one(game, eval_a, eval_b, args.sims, args.sims, args.max_moves, args.seed + i)
            ra = r
        else:
            r, _, _ = play_one(game, eval_b, eval_a, args.sims, args.sims, args.max_moves, args.seed + i)
            ra = 1.0 - r
        wins += ra
        if ra == 1.0:
            nw += 1
        elif ra == 0.0:
            nl += 1
        else:
            nd += 1
    dt = time.perf_counter() - t0
    elo, err = elo_from_score(wins, args.games)
    from tools.match.go_sprt import sprt_bounds, sprt_decide, sprt_llr
    llr = sprt_llr(nw, nl, nd, args.elo0, args.elo1)
    lower, upper = sprt_bounds()
    decision = sprt_decide(llr, lower, upper)
    print(f"games={args.games} score={wins:.1f} elo={elo:.1f}+-{err:.1f} seconds={dt:.2f}")
    print(f"w={nw} l={nl} d={nd} llr={llr:.3f} bounds=[{lower:.3f},{upper:.3f}] sprt={decision}")


if __name__ == "__main__":
    main()
