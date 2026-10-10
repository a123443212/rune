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

import numpy as np

from tools.analysis.oracle_routing import bucket_of


def imbalance_of(fen):
    from training.features.python_features import parse_fen

    vals = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}
    board, _, _, _ = parse_fen(fen)
    w = sum(vals[c[0]] for c in board if c is not None and c[1] == "w")
    b = sum(vals[c[0]] for c in board if c is not None and c[1] == "b")
    return "imbalanced" if abs(w - b) >= 3 else "balanced"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--preds", default="")
    ap.add_argument("--bench", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    legs = []
    for d in args.runs:
        with open(os.path.join(d, "metrics.json")) as f:
            m = json.load(f)
        val = m.get("val", {})
        legs.append({"model": m.get("model"), "architecture": m.get("architecture"),
                     "trained_positions": m.get("trained_positions"),
                     "val_total": val.get("total"), "wdl_acc": val.get("wdl_acc"),
                     "rank_acc": val.get("rank_acc"),
                     "student_vs_teacher_mae": val.get("student_vs_teacher_mae"),
                     "teacher_vs_target_mae": val.get("teacher_vs_target_mae"),
                     "params": val.get("params"),
                     "teacher_id": m.get("teacher_id"),
                     "distillation": m.get("distillation", {})})
    bench = {}
    if args.bench:
        with open(args.bench) as f:
            bench = json.load(f)
    for leg in legs:
        leg.update(bench.get(leg["model"], {}))

    frontier = sorted(
        [l for l in legs if l.get("val_total") is not None],
        key=lambda l: (l.get("params") or 0))
    report = {"legs": legs, "frontier_order": [l["model"] for l in frontier],
              "warning": "rejected levels are results; holes in the frontier are honest"}

    if args.preds:
        preds = [json.loads(l) for l in open(args.preds) if l.strip()]
        buckets = {}
        for r in preds:
            for key in ("teacher_value", "pred_value"):
                if key not in r:
                    raise ValueError(f"pred record missing {key}")
            err = abs(r["teacher_value"] - r["pred_value"])
            labels = dict(bucket_of(r))
            labels["imbalance"] = imbalance_of(r["fen"])
            for name, val in labels.items():
                d = buckets.setdefault(f"{name}={val}",
                                       {"n": 0, "mae": 0.0, "worst": 0.0})
                d["n"] += 1
                d["mae"] += err
                d["worst"] = max(d["worst"], err)
        for d in buckets.values():
            d["mae"] /= max(1, d["n"])
        report["retention"] = buckets
        worst = sorted(buckets.items(), key=lambda kv: kv[1]["mae"], reverse=True)[:5]
        report["fails_first"] = [{"bucket": k, **v} for k, v in worst]
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"legs": len(legs),
                      "frontier_order": report["frontier_order"],
                      "fails_first": report.get("fails_first", [])}, indent=2))


if __name__ == "__main__":
    main()
