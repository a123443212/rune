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
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def material_bucket(fen):
    board = fen.split(" ")[0]
    pieces = sum(1 for c in board if c.isalpha())
    if pieces <= 8:
        return "endgame-sparse"
    if pieces <= 16:
        return "endgame"
    if pieces <= 24:
        return "middlegame"
    return "opening"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    pool = []
    for p in args.positions:
        pool += [l.strip() for l in open(p) if l.strip()]
    rng.shuffle(pool)
    pool = pool[:args.max_n]
    recs = []
    for fen in pool:
        recs.append({"fen": fen, "material": material_bucket(fen), "phase": material_bucket(fen), "source": "search-native-sample"})
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print("wrote %d search-native positions to %s" % (len(recs), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
