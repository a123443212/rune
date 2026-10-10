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
def gate_fn(name, s):
    if name == "hard_sigmoid":
        return np.clip(0.2 * s + 0.5, 0, 1)
    if name == "screlu":
        c = np.clip(s, 0, 1)
        return c * c
    if name != "clip":
        raise ValueError(f"unknown gate {name}")
    return np.clip(s, 0, 1)


def py_value(model, fen):
    header, arrays = load_exported_arrays(model)
    if int(header.get("head_buckets", 1)) != 1:
        raise ValueError("cross_check supports head_buckets=1 only")
    if bool(header.get("head_pair", False)):
        raise ValueError("cross_check supports head_pair=false only")
    if header.get("game", "chess") != "chess":
        raise ValueError("cross_check supports chess only")
    feats = extract_features(fen)
    acc = np.zeros((8, 32), dtype=np.float64)
    for g, i in feats:
        t = 0 if g == 8 else g
        acc[t] += arrays["emb" + str(g)][i]
    tok = np.clip(acc, 0, 1).astype(np.float32)
    arch = header.get("architecture_id", header.get("arch"))
    if arch in ("RUNE-ATTN", "RUNE-ATTN-GAB"):
        Q = tok @ arrays["wq"].T + arrays["bq"]
        K = tok @ arrays["wk"].T + arrays["bk"]
        V = tok @ arrays["wvv"].T + arrays["bvv"]
        S = Q @ K.T + arrays["gab"]
        mixed = tok + gate_fn(header.get("gate", "clip"), S) @ V
    elif arch == "RUNE-REL-02":
        if header.get("geometric_bias") == "dynamic":
            raise RuntimeError("dynamic relational bias needs board context")
        Q = tok @ arrays["wq"].T + arrays["bq"]
        K = tok @ arrays["wk"].T + arrays["bk"]
        V = tok @ arrays["wvv"].T + arrays["bvv"]
        gab = arrays.get("gabS", arrays.get("gab"))
        S = Q @ K.T + gab
        G = gate_fn(header.get("gate", "clip"), S)
        alpha = float(header.get("alpha", 1.0))
        mixed = tok + alpha * (G @ V)
    elif arch == "RUNE-ATTN-DUAL":
        q1 = tok @ arrays["wq"].T + arrays["bq"]
        k1 = tok @ arrays["wk"].T + arrays["bk"]
        v1 = tok @ arrays["wvv"].T + arrays["bvv"]
        mid = tok + gate_fn(header.get("gate", "clip"), q1 @ k1.T + arrays["gab"]) @ v1
        q2 = mid @ arrays["wq2"].T + arrays["bq2"]
        k2 = mid @ arrays["wk2"].T + arrays["bk2"]
        v2 = mid @ arrays["wvv2"].T + arrays["bvv2"]
        mixed = mid + gate_fn(header.get("gate", "clip"), q2 @ k2.T + arrays["gab2"]) @ v2
    elif arch == "RUNE-ATTN-MH4":
        parts = []
        for h in range(4):
            sfx = f"_h{h}"
            Q = tok @ arrays["wq" + sfx].T + arrays["bq" + sfx]
            K = tok @ arrays["wk" + sfx].T + arrays["bk" + sfx]
            V = tok @ arrays["wv" + sfx].T + arrays["bv" + sfx]
            S = Q @ K.T + arrays["gab" + sfx]
            parts.append(gate_fn(header.get("gate", "clip"), S) @ V)
        cat = np.concatenate(parts, axis=-1)
        mixed = tok + cat @ arrays["wo"].T + arrays["bwo"]
    elif arch in ("RUNE-SFNN", "RUNE-MLP", "RUNE-MLP-S", "RUNE-SFNN-C"):
        mixed = tok
    else:
        raise ValueError(f"cross_check unsupported arch {arch}")
    flat = mixed.reshape(-1)
    h1 = np.clip(flat @ arrays["w1"].T + arrays["b1"], 0, 1)
    h2 = np.clip(h1 @ arrays["w2"].T + arrays["b2"], 0, 1)
    wvo = arrays.get("wvo", arrays.get("wv"))
    bvo = arrays.get("bvo", arrays.get("bv"))
    v = float(np.tanh(h2 @ wvo.T + bvo).reshape(-1)[0])
    wdl = (h2 @ arrays["wwdl"].T + arrays["bwdl"]).reshape(-1).tolist()
    return v, wdl
def run_cpp(cpp_bin, model, fen, path="auto"):
    out = subprocess.check_output([cpp_bin, "--model", model, "--fen", fen, "--path", path], text=True)
    parts = out.strip().split()
    return float(parts[0]), [float(x) for x in parts[1:4]]
def run_rust(rust_bin, model, fen, kernel="auto"):
    out = subprocess.check_output([rust_bin, "eval", "--model", model, "--fen", fen, "--kernel", kernel], text=True)
    v = None
    w = None
    for line in out.splitlines():
        if line.startswith("value"):
            v = float(line.split()[1])
        if line.startswith("wdl"):
            w = [float(x) for x in line.split()[1:4]]
    return v, w
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--cpp-bin", default="build/rune_eval")
    ap.add_argument("--rust-bin", default="target/debug/rune")
    ap.add_argument("--tol", type=float, default=2e-5)
    ap.add_argument("--cpp-path", default="auto")
    ap.add_argument("--rust-kernel", default="auto")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    with open(args.positions) as f:
        fens = [l.strip() for l in f if l.strip()]
    rows = []
    ok = True
    for fen in fens:
        pv, pw = py_value(args.model, fen)
        try:
            cv, cw = run_cpp(args.cpp_bin, args.model, fen, args.cpp_path)
        except Exception as e:
            cv, cw = float("nan"), [float("nan")] * 3
            print("cpp failed " + str(e))
        try:
            rv, rw = run_rust(args.rust_bin, args.model, fen, args.rust_kernel)
        except Exception as e:
            rv, rw = float("nan"), [float("nan")] * 3
            print("rust failed " + str(e))
        d_pc = abs(pv - cv) if pv == pv and cv == cv else float("inf")
        d_pr = abs(pv - rv) if pv == pv and rv == rv else float("inf")
        d_cr = abs(cv - rv) if cv == cv and rv == rv else float("inf")
        row = {"fen": fen, "py": pv, "cpp": cv, "rust": rv, "py_cpp": d_pc, "py_rust": d_pr, "cpp_rust": d_cr, "pass": d_pc <= args.tol and d_pr <= args.tol and d_cr <= args.tol}
        rows.append(row)
        if not row["pass"]:
            ok = False
        print(fen[:40] + " py=%.6f cpp=%.6f rust=%.6f d_cr=%.2g %s" % (pv, cv, rv, d_cr, "PASS" if row["pass"] else "FAIL"))
    rep = {"model": args.model, "tol": args.tol, "rows": rows, "pass": ok}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
    sys.exit(0 if ok else 2)
if __name__ == "__main__":
    main()
