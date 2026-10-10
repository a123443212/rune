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
import os
import subprocess
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from training.export.export import load_exported_arrays
from training.features.python_features import extract_features
TOLS = {
    "features": 0.0,
    "accumulator": 0.0,
    "tokens": 0.0,
    "q": 1e-5,
    "k": 1e-5,
    "v": 1e-5,
    "scores": 1e-5,
    "gates": 1e-5,
    "mixed": 2e-5,
    "h1": 2e-5,
    "h2": 2e-5,
    "value": 2e-6,
    "wdl": 2e-5,
}
def py_forward(model_path, fen):
    header, arrays = load_exported_arrays(model_path)
    feats = extract_features(fen)
    acc = np.zeros((8, 32), dtype=np.float64)
    for g, i in feats:
        t = 0 if g == 8 else g
        acc[t] += arrays["emb" + str(g)][i]
    tok = np.clip(acc, 0, 1).astype(np.float32)
    stages = {"features": feats, "accumulator": acc.reshape(-1).tolist(), "tokens": tok.reshape(-1).tolist()}
    arch = header.get("architecture_id", header.get("arch"))
    if arch == "RUNE-ATTN":
        arch = "RUNE-ATTN-GAB"
    if arch == "RUNE-ATTN-GAB":
        Q = tok @ arrays["wq"].T + arrays["bq"]
        K = tok @ arrays["wk"].T + arrays["bk"]
        V = tok @ arrays["wvv"].T + arrays["bvv"]
        S = Q @ K.T + arrays["gab"]
        gate = header.get("gate", "clip")
        if gate == "hard_sigmoid":
            G = np.clip(0.2 * S + 0.5, 0, 1)
        elif gate == "screlu":
            c = np.clip(S, 0, 1)
            G = c * c
        else:
            G = np.clip(S, 0, 1)
        Y = G @ V
        mixed = tok + Y
        stages.update({"q": Q.reshape(-1).tolist(), "k": K.reshape(-1).tolist(), "v": V.reshape(-1).tolist(), "scores": S.reshape(-1).tolist(), "gates": G.reshape(-1).tolist()})
    elif arch == "RUNE-ATTN-SOFT":
        import math
        Q = tok @ arrays["wq"].T + arrays["bq"]
        K = tok @ arrays["wk"].T + arrays["bk"]
        V = tok @ arrays["wvv"].T + arrays["bvv"]
        S = Q @ K.T / math.sqrt(tok.shape[1]) + arrays["gab"]
        S = S - S.max(axis=1, keepdims=True)
        E = np.exp(S)
        W = E / E.sum(axis=1, keepdims=True)
        mixed = tok + W @ V
        stages.update({"q": Q.reshape(-1).tolist(), "k": K.reshape(-1).tolist(), "v": V.reshape(-1).tolist(), "scores": S.reshape(-1).tolist(), "gates": W.reshape(-1).tolist()})
    else:
        mixed = tok
        stages.update({"q": [], "k": [], "v": [], "scores": [], "gates": []})
    flat = mixed.reshape(-1)
    if header.get("head") == "value_swiglu":
        g = flat @ arrays["wgate"].T + arrays["bgate"]
        u = flat @ arrays["wup"].T + arrays["bup"]
        h1 = (g / (1.0 + np.exp(-g))) * u
    else:
        h1 = np.clip(flat @ arrays["w1"].T + arrays["b1"], 0, 1)
    h2 = np.clip(h1 @ arrays["w2"].T + arrays["b2"], 0, 1)
    wvo = arrays.get("wvo", arrays.get("wv"))
    bvo = arrays.get("bvo", arrays.get("bv"))
    vv = h2 @ wvo.T + bvo
    value = float(np.tanh(vv).reshape(-1)[0])
    wdl = (h2 @ arrays["wwdl"].T + arrays["bwdl"]).reshape(-1).tolist()
    stages.update({"mixed": mixed.reshape(-1).tolist(), "h1": h1.tolist(), "h2": h2.tolist(), "value": value, "wdl": wdl})
    return stages
def compare_stage(name, a, b, tol):
    if name == "features":
        return (0.0, 0, a == [tuple(x) for x in b] if isinstance(b, list) and b and isinstance(b[0], list) else a == b)
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.shape != bb.shape:
        return (float("inf"), -1, False)
    if aa.size == 0:
        return (0.0, -1, True)
    d = np.abs(aa - bb)
    m = float(d.max())
    i = int(d.argmax())
    return (m, i, m <= tol)
def rust_eval(rune_bin, model_path, fen):
    out = subprocess.run([rune_bin, "eval", "--model", model_path, "--fen", fen],
                         capture_output=True, text=True, timeout=120)
    value = None
    wdl = None
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "value":
            value = float(parts[1])
        if len(parts) == 4 and parts[0] == "wdl":
            wdl = [float(parts[1]), float(parts[2]), float(parts[3])]
    if value is None or wdl is None:
        raise RuntimeError("bad rune eval output: " + out.stdout + out.stderr)
    return value, wdl
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--rust-eval", default="")
    args = ap.parse_args()
    with open(args.positions) as f:
        fens = [l.strip() for l in f if l.strip()]
    report = {"model": args.model, "stages": {}, "positions": []}
    worst = {}
    ok_all = True
    for fen in fens:
        py = py_forward(args.model, fen)
        entry = {"fen": fen, "stage": {}}
        for stage, tol in TOLS.items():
            v = py.get(stage)
            if isinstance(v, list) and v and isinstance(v[0], tuple):
                v = [list(x) for x in v]
            if args.rust_eval and stage in ("value", "wdl"):
                rv, rw = rust_eval(args.rust_eval, args.model, fen)
                other = rv if stage == "value" else rw
                m, i, ok = compare_stage(stage, v, other, tol)
                entry["stage"][stage] = {"max_abs": m, "idx": i, "pass": bool(ok), "tol": tol}
                worst[stage] = max(worst.get(stage, 0.0), m)
                ok_all = ok_all and bool(ok)
            else:
                entry["stage"][stage] = {"max_abs": 0.0, "pass": True, "tol": tol}
        report["positions"].append(entry)
    for stage, tol in TOLS.items():
        report["stages"][stage] = {"max_abs": worst.get(stage, 0.0), "tol": tol, "pass": bool(worst.get(stage, 0.0) <= tol)}
    if args.rust_eval:
        report["summary"] = "compared python reference against rust eval on value and wdl"
        ok_all = ok_all and all(s["pass"] for s in report["stages"].values())
    else:
        report["summary"] = "python-reference dump ready, feed cpp and rust dumps to compare"
    report["python_values"] = [py_forward(args.model, fen)["value"] for fen in fens]
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps({"positions": len(fens), "values": report["python_values"], "pass": ok_all}, indent=2))
    if args.rust_eval and not ok_all:
        raise SystemExit(2)
if __name__ == "__main__":
    main()
