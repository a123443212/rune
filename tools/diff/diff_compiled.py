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
from training.compiler.reference import forward_generic
from training.export.export import load_exported_arrays


def stage_report(arrays, tokens, dim, h1, gate, alpha, flat, eps):
    r = forward_generic(arrays, tokens, dim, h1, gate, alpha, flat)
    lines = []
    lines.append("tokens %d dim %d h1 %d" % (tokens, dim, h1))
    lines.append("value %.9f" % r["value"])
    lines.append("wdl %.6f %.6f %.6f" % (r["wdl"][0], r["wdl"][1], r["wdl"][2]))
    lines.append("q_mean %.6f k_mean %.6f v_mean %.6f" % (r["q"].mean(), r["k"].mean(), r["v"].mean()))
    lines.append("scores_max %.6f gate_mean %.6f" % (r["scores"].max(), r["gate"].mean()))
    return r, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generic", required=True)
    ap.add_argument("--compiled", required=True)
    ap.add_argument("--tol", type=float, default=2e-5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    hg, ag = load_exported_arrays(args.generic)
    hc, ac = load_exported_arrays(args.compiled)
    tokens = int(hg.get("tokens", 8))
    dim = int(hg.get("token_dim", 32))
    h1 = int(hg.get("head_h1", 128))
    gate = hg.get("gate", "clip")
    alpha = float(hg.get("alpha", 1.0))
    rng = np.random.RandomState(0)
    maxd = 0.0
    rep = {"generic": args.generic, "compiled": args.compiled, "stages": []}
    for trial in range(5):
        flat = np.clip(rng.randn(tokens * dim).astype(np.float32) * 0.3, 0, 1)
        r1 = forward_generic(ag, tokens, dim, h1, gate, alpha, flat)
        try:
            r2 = forward_generic(ac, tokens, dim, h1, gate, alpha, flat)
        except KeyError as e:
            print("compiled missing tensor %s" % e)
            return 2
        for key in ("q", "k", "v", "scores", "gate", "mixed", "h1", "h2"):
            d = float(np.abs(r1[key] - r2[key]).max())
            rep["stages"].append({"trial": trial, "stage": key, "max_abs": d})
            print("trial %d stage %s max_abs %.2e" % (trial, key, d))
            maxd = max(maxd, d)
        dv = abs(float(r1["value"]) - float(r2["value"]))
        dw = float(np.abs(r1["wdl"] - r2["wdl"]).max())
        rep["stages"].append({"trial": trial, "stage": "value", "max_abs": dv})
        rep["stages"].append({"trial": trial, "stage": "wdl", "max_abs": dw})
        print("trial %d value %.6f vs %.6f diff %.2e wdl_diff %.2e" % (trial, r1["value"], r2["value"], dv, dw))
        maxd = max(maxd, dv, dw)
    status = "PASS" if maxd <= args.tol else "FAIL"
    print("max_abs_diff %.9f tol %.9f %s" % (maxd, args.tol, status))
    rep["max_abs_diff"] = maxd
    rep["tol"] = args.tol
    rep["status"] = status
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
