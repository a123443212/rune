import sys

import torch

sys.path.insert(0, ".")

from tests.test_v13_incremental import MOVE_PAIRS
from training.features.python_features import NUM_GROUPS, extract_features
from training.rune_v13 import IncrementalRelationalState, build_dense_graph, group_deltas_to_tokens


def run_sequence(fens, tables, st, dim=32):
    g = build_dense_graph(8)
    prev_feats = extract_features(fens[0])
    prev = group_deltas_to_tokens(tables, prev_feats, prev_feats, dim)["tok_new"]
    st.rebuild(prev)
    for depth, fen in enumerate(fens[1:]):
        feats = extract_features(fen)
        r = group_deltas_to_tokens(tables, prev_feats, feats, dim)
        cur = r["tok_new"]
        ct = r["changed_tokens"]
        out = st.update(cur, ct)
        ok, diffs = st.verify_against_full(cur, tol=1e-5)
        cells = len(g.affected_edges(ct))
        status = "ok" if ok else "DIVERGED"
        print(f"node={depth} changed_tokens={ct} cells={cells}/64 worst={max(diffs.values()):.2e} {status}")
        if not ok:
            print(f"  position={fen}")
            print(f"  full_vs_incr={diffs}")
            return False
        prev_feats = feats
    return True


def run_branch(fens, tables, w):
    torch.manual_seed(0)
    dim = 32
    st = IncrementalRelationalState(
        w["wq"], w["bq"], w["wk"], w["bk"], w["wv"], w["bv"], w["gab"], threshold=8
    )
    r0 = group_deltas_to_tokens(tables, extract_features(fens[0]), extract_features(fens[0]), dim)
    st.rebuild(r0["tok_new"])
    outs = {}
    for name in ("quiet", "capture", "castle"):
        fa, fb = MOVE_PAIRS[name]
        base = extract_features(fa)
        nxt = extract_features(fb)
        rb = group_deltas_to_tokens(tables, base, base, dim)
        st.rebuild(rb["tok_new"])
        rn = group_deltas_to_tokens(tables, base, nxt, dim)
        st.push()
        st.update(rn["tok_new"], rn["changed_tokens"])
        ok, _ = st.verify_against_full(rn["tok_new"], tol=1e-5)
        outs[name] = (ok, st.cache["out"].clone())
        st.pop()
    for name, (ok, _) in outs.items():
        print(f"branch={name} parity={ok}")
    assert outs["quiet"][1] is not outs["capture"][1]
    assert not torch.equal(outs["quiet"][1], outs["capture"][1])
    print("branch isolation ok")


def main():
    torch.manual_seed(3)
    tables = [torch.randn(512, 32) * 0.01 for _ in range(NUM_GROUPS)]
    torch.manual_seed(7)
    w = {
        "wq": torch.randn(32, 32) * 0.08,
        "bq": torch.randn(32) * 0.01,
        "wk": torch.randn(32, 32) * 0.08,
        "bk": torch.randn(32) * 0.01,
        "wv": torch.randn(32, 32) * 0.08,
        "bv": torch.randn(32) * 0.01,
        "gab": torch.zeros(8, 8),
    }
    st = IncrementalRelationalState(
        w["wq"], w["bq"], w["wk"], w["bk"], w["wv"], w["bv"], w["gab"], threshold=8
    )
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    ]
    print("== sequence ==")
    run_sequence(fens, tables, st)
    print("== branches ==")
    run_branch(fens, tables, w)


if __name__ == "__main__":
    main()
