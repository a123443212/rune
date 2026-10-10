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
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.analysis.oracle_routing import bucket_of


def imbalance_of(fen):
    from training.features.python_features import parse_fen

    vals = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}
    board, _, _, _ = parse_fen(fen)
    w = sum(vals[c[0]] for c in board if c is not None and c[1] == "w")
    b = sum(vals[c[0]] for c in board if c is not None and c[1] == "b")
    return "imbalanced" if abs(w - b) >= 3 else "balanced"


def summarize(recs, teacher_key="teacher_value"):
    total = len(recs)
    axes = {}
    for r in recs:
        labels = dict(bucket_of({"fen": r["fen"], "teacher_value": r.get(teacher_key, 0.0)}))
        labels["imbalance"] = imbalance_of(r["fen"])
        for name, val in labels.items():
            axes.setdefault(name, {}).setdefault(val, 0)
            axes[name][val] += 1
    cov = {}
    for name, dist in axes.items():
        cov[name] = {k: {"n": v, "frac": v / max(1, total)} for k, v in dist.items()}
    return {"n": total, "coverage": cov}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def load_fens(path):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_bridge"))
        from reader import load_dataset

        try:
            _, _, recs = load_dataset(path)
            return [{"fen": r["fen"]} for r in recs]
        except Exception:
            pass
        return [{"fen": json.loads(l)["fen"]} for l in open(path) if l.strip()]

    pool = load_fens(args.pool)
    data = load_fens(args.dataset)
    batch = load_fens(args.batch)
    report = {
        "pool": summarize(pool),
        "dataset": summarize(data),
        "batch": summarize(batch),
        "warning": "watch batch vs pool drift: tactical/phase collapse is a kill signal",
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    for name, part in (("pool", report["pool"]), ("dataset", report["dataset"]),
                       ("batch", report["batch"])):
        print(f"== {name} n={part['n']}")
        for ax, dist in part["coverage"].items():
            print(f"  {ax}: " + ", ".join(f"{k}={v['frac']:.2f}" for k, v in dist.items()))


if __name__ == "__main__":
    main()
