import random

CHILD_CAP = 64


def enumerate_children(rb, parent_fen, cap=CHILD_CAP):
    b = rb.Board(parent_fen)
    out = []
    for m in b.legal_moves():
        if len(out) >= cap:
            break
        mv = rb.Move()
        mv.from_sq, mv.to_sq, mv.promo = m[0], m[1], m[2]
        if not b.make_move(mv):
            continue
        out.append(b.to_fen())
        b.unmake_move()
    return out


def build_sibling_pairs(rb, parent_fens, teacher_fn, teacher_id, max_pairs=1000,
                        margin_min=0.1, seed=0):
    rng = random.Random(seed)
    pairs = []
    cands = []
    for parent in parent_fens:
        try:
            children = enumerate_children(rb, parent)
        except Exception:
            continue
        if len(children) < 2:
            continue
        scored = [(c, teacher_fn(c)) for c in children]
        for i in range(len(scored)):
            for j in range(i + 1, len(scored)):
                (ca, va), (cb, vb) = scored[i], scored[j]
                if abs(va - vb) < margin_min:
                    continue
                sign = 1.0 if va >= vb else -1.0
                cands.append({"parent": parent, "child_a": ca, "child_b": cb,
                              "sign": sign, "teacher_a": va, "teacher_b": vb,
                              "teacher_id": teacher_id, "margin": abs(va - vb)})
    rng.shuffle(cands)
    return cands[:max_pairs]


def pairs_to_batch(dataset, pairs):
    fens = []
    for p in pairs:
        fens.append(p["child_a"])
        fens.append(p["child_b"])
    recs = [{"fen": f, "value": 0.0, "wdl": 1} for f in fens]
    sub = type(dataset)(recs, max_per_group=dataset.max_per_group)
    loader_ids = sub.collate(list(range(len(recs))))
    ids, masks = loader_ids[0], loader_ids[1]
    n = len(pairs)
    a_idx = list(range(0, 2 * n, 2))
    b_idx = list(range(1, 2 * n, 2))
    signs = [p["sign"] for p in pairs]
    weights = [min(1.0, p["margin"] * 2.0) for p in pairs]
    return ids, masks, a_idx, b_idx, signs, weights
