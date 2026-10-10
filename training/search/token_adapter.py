import torch


def feats_to_tensors(game, feats):
    ids = []
    masks = []
    for g in range(game.num_groups):
        idx = [i for q, i in feats if q == g]
        if idx:
            ids.append(torch.tensor([idx]))
            masks.append(torch.ones(1, len(idx)))
        else:
            ids.append(torch.zeros(1, 1, dtype=torch.long))
            masks.append(torch.zeros(1, 1))
    return ids, masks


def uniform_priors(legal):
    k = float(len(legal))
    return [1.0 / k for _ in legal]


def build_token_callbacks(game, model, legal_fn):
    def eval_fn(state):
        feats = game.extract(state)
        ids, masks = feats_to_tensors(game, feats)
        with torch.no_grad():
            v, _ = model(ids, masks)
        value = float(v[0])
        if value > 1.0:
            value = 1.0
        if value < -1.0:
            value = -1.0
        legal = legal_fn(state)
        return value, uniform_priors(legal)

    return eval_fn


def search_move(game, model, state, legal_fn, apply_fn, simulations=32, seed=1):
    from training.search.mcts import PuctSearch
    eval_fn = build_token_callbacks(game, model, legal_fn)
    search = PuctSearch(simulations=simulations, seed=seed)
    return search.search(state, legal_fn, apply_fn, eval_fn)
