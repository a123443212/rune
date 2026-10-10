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
import itertools
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.preds) if l.strip()]
    fens = {}
    for r in recs:
        fens.setdefault(r.get("parent", ""), []).append(r)
    pairs = 0
    rank_err = 0
    delta_err = 0.0
    for parent, kids in fens.items():
        if len(kids) < 2 or parent == "":
            continue
        for a, b in itertools.combinations(kids, 2):
            td = float(a["teacher_value"]) - float(b["teacher_value"])
            pd = float(a["pred_value"]) - float(b["pred_value"])
            pairs += 1
            delta_err += abs(td - pd)
            if (td > 0) != (pd > 0) and abs(td) > 0.02:
                rank_err += 1
    out = {"pairs": pairs, "rank_err": rank_err, "rank_err_rate": rank_err / max(1, pairs), "mean_delta_err": delta_err / max(1, pairs)}
    print(json.dumps(out, indent=2))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
