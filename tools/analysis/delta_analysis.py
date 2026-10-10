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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tactical-weight", type=float, default=2.0)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.pairs) if l.strip()]
    for r in recs:
        for key in ("fen_a", "fen_b", "teacher_a", "teacher_b",
                    "student_a", "student_b"):
            if key not in r:
                raise ValueError(f"pair record missing {key}")
    if not recs:
        raise ValueError("empty pairs file")

    dt = np.array([r["teacher_b"] - r["teacher_a"] for r in recs])
    ds = np.array([r["student_b"] - r["student_a"] for r in recs])
    tact = np.array([1.0 if r.get("tactical") else 0.0 for r in recs])
    w = 1.0 + (args.tactical_weight - 1.0) * tact
    abs_err = np.abs(dt - ds)
    sign_flip = np.sign(dt) != np.sign(ds)
    sharp = np.abs(dt) > 0.3

    report = {
        "n": len(recs),
        "warning": "student must preserve deltas, not just absolute scalars",
        "delta_mae": float(abs_err.mean()),
        "delta_mae_weighted": float((abs_err * w).sum() / w.sum()),
        "sign_flip_rate": float(sign_flip.mean()),
        "sign_flip_rate_weighted": float((sign_flip * w).sum() / w.sum()),
        "sharp_move_delta_mae": float(abs_err[sharp].mean()) if sharp.sum() else 0.0,
        "sharp_n": int(sharp.sum()),
        "corr_teacher_student_delta": float(np.corrcoef(dt, ds)[0, 1])
        if len(recs) > 2 and dt.std() > 0 and ds.std() > 0 else 0.0,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
