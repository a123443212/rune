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


def ranks(a):
    order = np.argsort(a)
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    return r


def spearman(a, b):
    ra, rb = ranks(np.asarray(a, float)), ranks(np.asarray(b, float))
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (ra * ra).sum() * (rb * rb).sum()
    if denom <= 0:
        return 0.0
    return float((ra * rb).sum() / denom ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--buckets", type=int, default=10)
    ap.add_argument("--need-tau", type=float, default=0.1)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.preds) if l.strip()]
    for r in recs:
        for key in ("fen", "teacher_value", "pred_value", "u"):
            if key not in r:
                raise ValueError(f"record missing {key}")
    if not recs:
        raise ValueError("empty predictions file")

    err = np.array([abs(r["teacher_value"] - r["pred_value"]) for r in recs])
    u = np.array([r["u"] for r in recs], dtype=float)
    order = np.argsort(u)
    n = len(recs)
    edges = [order[len(order) * i // args.buckets:len(order) * (i + 1) // args.buckets]
             for i in range(args.buckets)]
    curve = [{"bucket": i, "n": len(e), "u_mean": float(u[e].mean()) if len(e) else 0.0,
              "err_mean": float(err[e].mean()) if len(e) else 0.0,
              "err_p90": float(np.quantile(err[e], 0.9)) if len(e) else 0.0}
             for i, e in enumerate(edges)]
    mono = all(curve[i]["err_mean"] <= curve[i + 1]["err_mean"] + 1e-12
               for i in range(len(curve) - 1))

    lo = u < 1 / 3
    md = (u >= 1 / 3) & (u < 2 / 3)
    hi = u >= 2 / 3
    strat = {name: {"n": int(m.sum()),
                    "err_mean": float(err[m].mean()) if m.sum() else 0.0}
             for name, m in (("low", lo), ("medium", md), ("high", hi))}

    need = err > args.need_tau
    routed = np.array([bool(r.get("routed", False)) for r in recs])
    by_bucket = {}
    for i, r in enumerate(recs):
        for name, val in bucket_of(r).items():
            key = f"{name}={val}"
            d = by_bucket.setdefault(key, {"n": 0, "fn": 0, "tp": 0,
                                           "err_mean": 0.0, "u_mean": 0.0})
            d["n"] += 1
            d["err_mean"] += float(err[i])
            d["u_mean"] += float(u[i])
            if need[i] and routed[i]:
                d["tp"] += 1
            elif need[i]:
                d["fn"] += 1
    for d in by_bucket.values():
        d["err_mean"] /= max(1, d["n"])
        d["u_mean"] /= max(1, d["n"])
        d["danger_miss_rate"] = d["fn"] / max(1, d["fn"] + d["tp"])

    report = {
        "n": n,
        "warning": "calibration before routing. Uncorrelated u must not drive refinement.",
        "spearman_u_err": spearman(u, err),
        "calibration_curve": curve,
        "curve_monotone": mono,
        "stratification": strat,
        "stratification_holds": strat["low"]["err_mean"] <= strat["medium"]["err_mean"] <=
        strat["high"]["err_mean"] if all(v["n"] for v in strat.values()) else False,
        "need_tau": args.need_tau,
        "overall_danger_miss_rate": float((need & ~routed).sum() / max(1, need.sum())),
        "by_bucket": by_bucket,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("n", "spearman_u_err", "curve_monotone",
                                            "stratification",
                                            "stratification_holds",
                                            "overall_danger_miss_rate")}, indent=2))


if __name__ == "__main__":
    main()
