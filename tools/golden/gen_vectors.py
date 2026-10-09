import json
import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from training.features import python_features as pf
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "spec", "test-vectors", "v10")
FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "8/P7/8/8/8/1k6/8/4K3 w - - 0 1",
    "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K2R b KQkq - 0 1",
    "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
    "6k1/8/8/8/8/8/8/K6R w - - 100 45",
]
MASK64 = (1 << 64) - 1
def lcg_init(seed):
    s = (seed * 2 + 1) & MASK64
    return s
def lcg_next(s):
    s = (s * 6364136223846793005 + 1442695040888963407) & MASK64
    return s, (s >> 33) / 4294967295.0
def make_table(seed, vocab, dim):
    s = lcg_init(seed)
    scale = 1.0 / vocab
    out = []
    for _ in range(vocab * dim):
        s, u = lcg_next(s)
        out.append((u - 0.5) * 2.0 * scale)
    return out
def clip01(x):
    if x != x:
        return x
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
def mat_vec(mat, vec, bias, rows, cols):
    out = [0.0] * rows
    for r in range(rows):
        a = bias[r] if bias is not None else 0.0
        base = r * cols
        for c in range(cols):
            a += mat[base + c] * vec[c]
        out[r] = a
    return out
def mat_mul_tt(a, b, m, n, k):
    out = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for t in range(k):
                s += a[i * k + t] * b[j * k + t]
            out[i * n + j] = s
    return out
def mat_mul(a, b, m, n, k):
    out = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for t in range(k):
                s += a[i * k + t] * b[t * n + j]
            out[i * n + j] = s
    return out
def quant_half_away(w, scale, bound):
    if not (scale > 0.0):
        return 0
    if w != w:
        return 0
    q = w / scale
    if q >= 0:
        r = math.floor(q + 0.5)
    else:
        r = math.ceil(q - 0.5)
    if r > bound:
        r = bound
    if r < -bound:
        r = -bound
    if w > 1e30:
        return bound
    if w < -1e30:
        return -bound
    return int(r)
def build():
    os.makedirs(OUT, exist_ok=True)
    feats = []
    for fen in FENS:
        fl = pf.extract_features(fen)
        feats.append({"fen": fen, "features": fl})
    with open(os.path.join(OUT, "features.json"), "w") as f:
        json.dump({"version": 1, "feature_version": pf.FEATURE_VERSION, "vectors": feats}, f, indent=2)
    tables = {}
    for g in range(9):
        tables[str(g)] = make_table(7, pf.VOCAB_SIZES[g], pf.TOKEN_DIM)
    acc_cases = []
    for entry in feats:
        fl = entry["features"]
        acc = [0.0] * (8 * pf.TOKEN_DIM)
        for (g, idx) in fl:
            base_t = tables[str(g)]
            t = 0 if g == 8 else g
            for d in range(pf.TOKEN_DIM):
                acc[t * pf.TOKEN_DIM + d] += base_t[idx * pf.TOKEN_DIM + d]
        tok = [clip01(v) for v in acc]
        acc_cases.append({"fen": entry["fen"], "accumulator": acc, "tokens": tok})
    with open(os.path.join(OUT, "accumulator.json"), "w") as f:
        json.dump({"version": 1, "tables_seed": 7, "vectors": acc_cases}, f)
    qarr = [0.0, 0.5, -0.5, 1.0, -1.0, 0.123, -0.987, 127.0, -127.0]
    scale = 0.05
    qexp = [quant_half_away(w, scale, 127) for w in qarr]
    q16exp = [quant_half_away(w, scale, 32767) for w in qarr]
    with open(os.path.join(OUT, "quant.json"), "w") as f:
        json.dump({"version": 1, "scale": scale, "inputs": qarr, "int8": qexp, "int16": q16exp}, f, indent=2)
    t, d = 8, 32
    s = 999
    def init_vec(n, sc):
        nonlocal_s = [s]
        return init_vec_with_seed(n, sc, nonlocal_s)
    def init_vec_with_seed(n, sc, box):
        out = []
        for _ in range(n):
            box[0] = (box[0] * 6364136223846793005 + 1442695040888963407) & MASK64
            u = (box[0] >> 33) / 4294967295.0
            out.append((u - 0.5) * 2.0 * sc)
        return out
    box = [4242]
    wq = init_vec_with_seed(d * d, 0.08, box)
    bq = init_vec_with_seed(d, 0.01, box)
    wk = init_vec_with_seed(d * d, 0.08, box)
    bk = init_vec_with_seed(d, 0.01, box)
    wv = init_vec_with_seed(d * d, 0.08, box)
    bv = init_vec_with_seed(d, 0.01, box)
    gab = [0.0] * (t * t)
    tok_in = acc_cases[0]["tokens"]
    q = []
    k = []
    v = []
    for i in range(t):
        xb = tok_in[i * d:(i + 1) * d]
        q += mat_vec(wq, xb, bq, d, d)
        k += mat_mul_tt(wq[:0], wq[:0], 0, 0, 1)[:0] if False else mat_vec(wk, xb, bk, d, d)
        v += mat_vec(wv, xb, bv, d, d)
    sc = mat_mul_tt(q, k, t, t, d)
    gated = [clip01(x + gab[i]) for i, x in enumerate(sc)]
    y = mat_mul(gated, v, t, d, t)
    mixed = [tok_in[i] + y[i] for i in range(t * d)]
    with open(os.path.join(OUT, "mixer.json"), "w") as f:
        json.dump({"version": 1, "tokens": t, "dim": d, "input": tok_in, "wq": wq, "bq": bq, "wk": wk, "bk": bk, "wvv": wv, "bvv": bv, "gab": gab, "q": q, "k": k, "v": v, "scores": sc, "gates": gated, "mixed": mixed}, f)
    w1 = init_vec_with_seed(128 * 256, 0.05, box)
    b1 = init_vec_with_seed(128, 0.01, box)
    w2 = init_vec_with_seed(32 * 128, 0.05, box)
    b2 = init_vec_with_seed(32, 0.01, box)
    wvo = init_vec_with_seed(32, 0.05, box)
    bvo = init_vec_with_seed(1, 0.01, box)
    wwdl = init_vec_with_seed(96, 0.05, box)
    bwdl = init_vec_with_seed(3, 0.01, box)
    h1 = [clip01(x) for x in mat_vec(w1, mixed, b1, 128, 256)]
    h2 = [clip01(x) for x in mat_vec(w2, h1, b2, 32, 128)]
    vv = bvo[0] + sum(wvo[i] * h2[i] for i in range(32))
    value = math.tanh(vv)
    wdl = mat_vec(wwdl, h2, bwdl, 3, 32)
    with open(os.path.join(OUT, "head.json"), "w") as f:
        json.dump({"version": 1, "mixed": mixed, "h1": h1, "h2": h2, "value": value, "wdl": wdl}, f)
    routing = []
    for thr in [0.2, 0.5, 0.8]:
        for score in [0.1, 0.5, 0.9, float("nan")]:
            if score != score:
                refine = True
            elif score >= thr:
                refine = True
            else:
                refine = False
            routing.append({"score": score if score == score else "nan", "threshold": thr, "refine": refine})
    with open(os.path.join(OUT, "routing.json"), "w") as f:
        json.dump({"version": 1, "vectors": routing}, f, indent=2)
    full = []
    for i, entry in enumerate(feats):
        full.append({"fen": entry["fen"], "features": entry["features"], "tokens": acc_cases[i]["tokens"]})
    with open(os.path.join(OUT, "full.json"), "w") as f:
        json.dump({"version": 1, "vectors": full}, f)
    manifest = {"version": 1, "files": ["features.json", "accumulator.json", "quant.json", "mixer.json", "head.json", "routing.json", "full.json"]}
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote vectors to " + OUT)
if __name__ == "__main__":
    build()
