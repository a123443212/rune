import torch

from training.features.python_features import NUM_GROUPS


def _bucket(feats):
    out = {}
    for g, i in feats:
        out.setdefault(int(g), set()).add(int(i))
    return out


def detect_changed_groups(feats_old, feats_new):
    old = _bucket(feats_old)
    new = _bucket(feats_new)
    changed = []
    added = []
    removed = []
    for g in range(NUM_GROUPS):
        a = old.get(g, set())
        b = new.get(g, set())
        add = sorted(b - a)
        rem = sorted(a - b)
        if add or rem:
            changed.append(g)
        added.append([(g, i) for i in add])
        removed.append([(g, i) for i in rem])
    return changed, added, removed


def _group_accum(tables, feats, dim):
    acc = torch.zeros(NUM_GROUPS, dim)
    for g, i in feats:
        acc[int(g)] += tables[int(g)][int(i)]
    return acc


def group_deltas_to_tokens(tables, feats_old, feats_new, dim, token_of_group=None):
    if token_of_group is None:
        token_of_group = list(range(NUM_GROUPS))
    n_tokens = max(token_of_group) + 1
    acc_old = _group_accum(tables, feats_old, dim)
    acc_new = _group_accum(tables, feats_new, dim)
    delta_acc = acc_new - acc_old
    tok_old = torch.zeros(n_tokens, dim)
    tok_new = torch.zeros(n_tokens, dim)
    for g in range(NUM_GROUPS):
        t = token_of_group[g]
        tok_old[t] += acc_old[g]
        tok_new[t] += acc_new[g]
    tok_old = torch.clamp(tok_old, 0.0, 1.0)
    tok_new = torch.clamp(tok_new, 0.0, 1.0)
    changed_tokens = sorted(
        t for t in range(n_tokens) if not torch.equal(tok_old[t], tok_new[t])
    )
    return {
        "acc_old": acc_old,
        "acc_new": acc_new,
        "delta_acc": delta_acc,
        "tok_old": tok_old,
        "tok_new": tok_new,
        "delta_tok": tok_new - tok_old,
        "changed_tokens": changed_tokens,
    }
