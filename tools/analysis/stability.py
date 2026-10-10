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

PIECE_VALUES = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}


def subset_labels(rec):
    from training.features.python_features import parse_fen

    labels = dict(bucket_of({"fen": rec["parent_fen"],
                             "teacher_value": rec.get("parent_value", 0.0)}))
    board, _, _, _ = parse_fen(rec["parent_fen"])
    w = sum(PIECE_VALUES[c[0]] for c in board if c is not None and c[1] == "w")
    b = sum(PIECE_VALUES[c[0]] for c in board if c is not None and c[1] == "b")
    labels["imbalance"] = "imbalanced" if abs(w - b) >= 3 else "balanced"
    labels["queenless"] = "queenless" if labels.get("queen") == "no_queen" else labels.get(
        "queen", "unknown")
    if "tactical" in rec:
        labels["tactical"] = str(bool(rec["tactical"]))
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.pairs) if l.strip()]
    if not recs:
        raise ValueError("empty pairs file")

    spreads, flips, spikes, seps = [], 0, 0, []
    sharp_big, sharp_n = 0.0, 0
    by_subset = {}
    n_routed_pairs = 0
    for r in recs:
        for key in ("parent_fen", "parent_value", "children"):
            if key not in r:
                raise ValueError(f"record missing {key}")
        kids = [c["value"] for c in r["children"]]
        if not kids:
            continue
        pv = r["parent_value"]
        spread = max(kids) - min(kids)
        spreads.append(spread)
        if len(kids) >= 2:
            s = sorted(kids, reverse=True)
            seps.append(s[0] - s[1])
        pr, kr = r.get("parent_routed", None), [c.get("routed", None) for c in r["children"]]
        if pr is not None and all(k is not None for k in kr):
            n_routed_pairs += 1
            if any(bool(k) != bool(pr) for k in kr):
                flips += 1
        pu = r.get("parent_u", None)
        ku = [c.get("u", None) for c in r["children"]]
        if pu is not None and any(k is not None for k in ku):
            for k in ku:
                if k is not None and abs(k - pu) > 0.5:
                    spikes += 1
                    break
        if str(r.get("tactical", "")) == "True":
            sharp_big += max(0.0, max(kids) - pv)
            sharp_n += 1
        for name, val in subset_labels(r).items():
            d = by_subset.setdefault(f"{name}={val}", {"n": 0, "spread": 0.0})
            d["n"] += 1
            d["spread"] += spread
    spreads = np.array(spreads)
    report = {
        "n": len(recs),
        "warning": "distinguish legitimate sharpness from instability; never smooth tactics away",
        "spread": {"mean": float(spreads.mean()), "p50": float(np.median(spreads)),
                   "p90": float(np.quantile(spreads, 0.9))},
        "best_second_separation_mean": float(np.mean(seps)) if seps else 0.0,
        "routing_flips": flips,
        "routing_flip_rate": flips / max(1, n_routed_pairs),
        "uncertainty_spikes": spikes,
        "tactical_upside_mean": sharp_big / max(1, sharp_n),
        "tactical_n": sharp_n,
        "by_subset": {k: {"n": v["n"], "spread_mean": v["spread"] / max(1, v["n"])}
                      for k, v in by_subset.items()},
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("n", "spread", "best_second_separation_mean",
                                            "routing_flips", "routing_flip_rate",
                                            "uncertainty_spikes",
                                            "tactical_upside_mean")}, indent=2))


if __name__ == "__main__":
    main()
