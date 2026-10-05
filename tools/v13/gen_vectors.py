import os

import torch

from training.rune_v13 import IncrementalRelationalState
from training.rune_v13.relational_reference import dense_forward

VEC_TOKENS = 4
VEC_DIM = 8
VEC_SEED = 1307


def build_case():
    torch.manual_seed(VEC_SEED)
    t = VEC_TOKENS
    d = VEC_DIM
    wq = torch.randn(d, d) * 0.08
    bq = torch.randn(d) * 0.01
    wk = torch.randn(d, d) * 0.08
    bk = torch.randn(d) * 0.01
    wv = torch.randn(d, d) * 0.08
    bv = torch.randn(d) * 0.01
    gab = torch.full((t, t), 0.05)
    x0 = torch.rand(t, d)
    x1 = x0.clone()
    x1[1] += 0.07
    x1[3] -= 0.04
    x1.clamp_(0.0, 1.0)
    changed = [1, 3]
    ref_full = dense_forward(x1, wq, bq, wk, bk, wv, bv, gab)
    st = IncrementalRelationalState(wq, bq, wk, bk, wv, bv, gab, threshold=8)
    st.rebuild(x0)
    ref_incr = st.update(x1, changed)
    return {
        "t": t, "d": d, "threshold": 2, "gate": "clip", "alpha": 1.0,
        "wq": wq, "bq": bq, "wk": wk, "bk": bk, "wv": wv, "bv": bv,
        "gab": gab, "x0": x0, "x1": x1, "changed": changed,
        "full": ref_full, "incr": ref_incr,
    }


def _line(vals):
    return " ".join(f"{float(v):.9f}" for v in vals)


def write_vectors(path):
    c = build_case()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [f'{c["t"]} {c["d"]} {c["threshold"]} {c["gate"]} {c["alpha"]}']
    for key in ("wq", "bq", "wk", "bk", "wv", "bv", "gab", "x0", "x1", "full", "incr"):
        rows.append(_line(c[key].reshape(-1).tolist()))
    rows.append(" ".join(str(i) for i in c["changed"]))
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")
    return c


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    out = os.path.normpath(os.path.join(root, "spec", "test-vectors", "v13", "incremental.txt"))
    c = write_vectors(out)
    print(f"wrote {out} t={c['t']} d={c['d']}")
