import math

import torch


def move_to_index(move, n):
    if int(move) == -1:
        return n * n
    return int(move)


def priors_for_legal(policy, legal, n):
    idx = [move_to_index(m, n) for m in legal]
    vals = [float(policy[i]) if 0 <= i < len(policy) else 0.0 for i in idx]
    s = sum(vals)
    if s <= 0.0:
        k = float(len(legal))
        return [1.0 / k for _ in legal]
    return [v / s for v in vals]


def build_go_callbacks(game, model):
    n = int(game.size)

    def legal_fn(state):
        return game.legal(state)

    def apply_fn(state, move):
        return game.apply(state, int(move))

    def eval_fn(state):
        planes = game.planes(state)
        with torch.no_grad():
            v, _, _, probs = model(torch.tensor(planes))
        value = float(v[0])
        if value > 1.0:
            value = 1.0
        if value < -1.0:
            value = -1.0
        legal = game.legal(state)
        priors = priors_for_legal(probs[0].tolist(), legal, n)
        return value, priors

    return legal_fn, apply_fn, eval_fn


def search_move(game, model, state, simulations=32, seed=1):
    from training.search.mcts import PuctSearch
    legal_fn, apply_fn, eval_fn = build_go_callbacks(game, model)
    search = PuctSearch(simulations=simulations, seed=seed)
    return search.search(state, legal_fn, apply_fn, eval_fn)
