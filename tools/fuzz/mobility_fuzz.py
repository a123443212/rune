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
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from training.features.python_features import extract_features
def run_cpp_dump(cpp_bin, fens, dump_dir):
    pos = os.path.join(dump_dir, "in.txt")
    with open(pos, "w") as f:
        for fen in fens:
            f.write(fen + "\n")
    r = subprocess.run([cpp_bin, "--positions", pos, "--dump-dir", dump_dir, "--mode", "tolerant"], capture_output=True, text=True)
    if r.returncode not in (0, 2):
        raise RuntimeError("cpp dump failed: " + r.stdout + r.stderr)
    out = []
    for i in range(len(fens)):
        feats = []
        with open(os.path.join(dump_dir, "pos_%03d.txt" % i)) as f:
            for line in f:
                p = line.split()
                if len(p) == 3 and p[0] == "f":
                    feats.append((int(p[1]), int(p[2])))
        out.append(feats)
    return out
def run_rust_inspect(rust_bin, fen):
    r = subprocess.run([rust_bin, "inspect", "--fen", fen], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("rust inspect failed: " + r.stderr)
    feats = []
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) == 3 and p[0] == "f":
            feats.append((int(p[1]), int(p[2])))
    return feats
def mobility_of(feats):
    for g, i in feats:
        if g == 6 and 384 <= i < 448:
            return (i - 384) % 32
    return None
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cpp-bin", default="build/rune_diff")
    ap.add_argument("--rust-bin", default="target/release/rune")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    with open(args.corpus) as f:
        fens = [l.strip() for l in f if l.strip()]
    if args.limit > 0:
        fens = fens[:args.limit]
    tmp = tempfile.mkdtemp(prefix="mobfuzz_cpp_")
    cpp_lists = run_cpp_dump(args.cpp_bin, fens, tmp)
    bad = []
    for n, fen in enumerate(fens):
        py = sorted(set(extract_features(fen)))
        cp = cpp_lists[n]
        try:
            ru = run_rust_inspect(args.rust_bin, fen)
        except RuntimeError as e:
            bad.append({"fen": fen, "error": "rust: " + str(e)})
            continue
        row = {"fen": fen, "n_py": len(py), "n_cpp": len(cp), "n_rust": len(ru),
               "mob_py": mobility_of(py), "mob_cpp": mobility_of(cp), "mob_rust": mobility_of(ru)}
        if py != cp or py != ru:
            row["py_eq_cpp"] = (py == cp)
            row["py_eq_rust"] = (py == ru)
            row["cpp_eq_rust"] = (cp == ru)
            sp, sc, sr = set(py), set(cp), set(ru)
            row["only_py"] = sorted(sp - sc - sr)[:8]
            row["only_cpp"] = sorted(sc - sp - sr)[:8]
            row["only_rust"] = sorted(sr - sp - sc)[:8]
            bad.append(row)
        if (n + 1) % 100 == 0:
            print("checked %d/%d mismatch %d" % (n + 1, len(fens), len(bad)))
    print("checked %d mismatch %d" % (len(fens), len(bad)))
    for b in bad[:10]:
        print(json.dumps(b))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"corpus": args.corpus, "checked": len(fens), "mismatch": len(bad), "rows": bad}, f, indent=2)
    sys.exit(0 if not bad else 2)
if __name__ == "__main__":
    main()
