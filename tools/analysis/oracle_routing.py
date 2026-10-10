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
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from training.features.python_features import game_phase, parse_fen


def wdl_entropy(w):
    w = np.asarray(w, dtype=float)
    w = w / max(1e-12, w.sum())
    w = np.clip(w, 1e-12, 1.0)
    return float(-(w * np.log(w)).sum())


def material_of(fen):
    board, _, _, _ = parse_fen(fen)
    total = sum(1 for c in board if c is not None)
    queens = sum(1 for c in board if c is not None and c[0] == "q")
    return total, queens


def bucket_of(rec):
    fen = rec["fen"]
    board, _, _, _ = parse_fen(fen)
    phase = ("opening", "middlegame", "endgame")[game_phase(board)]
    total, queens = material_of(fen)
    buckets = {
        "phase": phase,
        "queen": "queen" if queens else "no_queen",
        "material": "heavy" if total > 20 else ("light" if total < 10 else "mid"),
        "boundary": "boundary" if abs(rec["teacher_value"]) < 0.15 else "decisive",
    }
    for key in ("tactical", "king_danger"):
        if key in rec:
            buckets[key] = str(bool(rec[key]))
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", default="0.05,0.1,0.2,0.3")
    ap.add_argument("--budgets", default="0.1,0.25,0.5,0.75")
    args = ap.parse_args()

    recs = []
    with open(args.preds) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    for r in recs:
        for key in ("fen", "teacher_value", "cheap_value", "cheap_wdl"):
            if key not in r:
                raise ValueError(f"record missing {key}: {r}")
    if not recs:
        raise ValueError("empty predictions file: teacher labels are required first")

    err = np.array([abs(r["teacher_value"] - r["cheap_value"]) for r in recs])
    ent = np.array([wdl_entropy(r["cheap_wdl"]) for r in recs])
    diff = np.array([r.get("difficulty", float("nan")) for r in recs])
    learned = diff if np.isfinite(diff).all() else ent
    learned_name = "difficulty" if np.isfinite(diff).all() else "wdl_entropy"

    taus = [float(x) for x in args.thresholds.split(",")]
    budgets = [float(x) for x in args.budgets.split(",")]
    order_learned = np.argsort(-learned)

    ceiling, gap = [], []
    for tau in taus:
        needs = err > tau
        base = float(err.mean())
        row = {"tau": tau, "mean_abs_err_cheap": base,
               "oracle_rate": float(needs.mean())}
        for b in budgets:
            k = max(1, int(len(recs) * b))
            top = np.zeros(len(recs), dtype=bool)
            top[order_learned[:k]] = True
            captured = float((needs & top).sum() / max(1, needs.sum()))
            residual = err[~top].mean() if (~top).sum() else 0.0
            row[f"budget_{b}"] = {"captured_need": captured,
                                 "residual_err": float(residual)}
        oracle_residual = err[~needs].mean() if (~needs).sum() else 0.0
        row["oracle_residual_err"] = float(oracle_residual)
        ceiling.append(row)
        gap.append({"tau": tau,
                    "oracle_rate": float(needs.mean()),
                    "learned_rate_for_95_capture": None})
        if needs.sum():
            hits = (needs[order_learned]).cumsum() / needs.sum()
            idx = int(np.searchsorted(hits, 0.95))
            gap[-1]["learned_rate_for_95_capture"] = float((idx + 1) / len(recs))

    fp_fn = {}
    tau0 = taus[len(taus) // 2]
    needs0 = err > tau0
    k0 = max(1, int(len(recs) * 0.25))
    routed = np.zeros(len(recs), dtype=bool)
    routed[order_learned[:k0]] = True
    for i, r in enumerate(recs):
        for name, val in bucket_of(r).items():
            key = f"{name}={val}"
            d = fp_fn.setdefault(key, {"n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0})
            d["n"] += 1
            hard, sent = bool(needs0[i]), bool(routed[i])
            d["tp" if hard and sent else "fp" if sent else "fn" if hard else "tn"] += 1
    for d in fp_fn.values():
        d["danger_miss_rate"] = d["fn"] / max(1, d["fn"] + d["tp"])

    report = {
        "n": len(recs),
        "learned_signal": learned_name,
        "warning": "ceiling only. No learned-routing investment without a ceiling.",
        "error_stats": {"mean": float(err.mean()), "p50": float(np.median(err)),
                        "p90": float(np.quantile(err, 0.9)),
                        "p99": float(np.quantile(err, 0.99))},
        "uncertainty_calibration": {
            "corr_entropy_error": float(np.corrcoef(ent, err)[0, 1])
            if len(recs) > 2 else 0.0,
        },
        "ceiling": ceiling,
        "learned_gap": gap,
        "confusion_by_bucket": fp_fn,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("n", "learned_signal", "error_stats",
                                            "uncertainty_calibration")}, indent=2))
    for row in ceiling:
        print(f"tau={row['tau']} oracle_rate={row['oracle_rate']:.3f} "
              f"oracle_residual={row['oracle_residual_err']:.4f}")


if __name__ == "__main__":
    main()
