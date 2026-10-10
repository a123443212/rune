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


def random_playout(board_cls, rng, max_plies=120):
    import rune_bindings as rb

    b = board_cls()
    fens = []
    for ply in range(max_plies):
        moves = b.legal_moves()
        if not moves:
            break
        fens.append((b.to_fen(), ply))
        m = moves[rng.randrange(len(moves))]
        mv = rb.Move()
        mv.from_sq = m[0]
        mv.to_sq = m[1]
        mv.promo = m[2]
        b.make_move(mv)
    return fens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build-dir", default="build")
    args = ap.parse_args()

    sys.path.insert(0, args.build_dir)
    import rune_bindings as rb

    from training.datasets import pipeline as P

    rng = random.Random(args.seed)
    records = []
    for g in range(args.games):
        for fen, ply in random_playout(rb.Board, rng):
            material = fen.split()[0]
            wq = material.count("Q")
            bq = material.count("q")
            score = (wq - bq) * 0.1 + rng.uniform(-0.2, 0.2)
            stm = fen.split()[1]
            if stm == "b":
                score = -score
            value = max(-1.0, min(1.0, score))
            wdl = 1 if abs(value) < 0.15 else (0 if value > 0 else 2)
            records.append(
                {
                    "fen": fen,
                    "value": value,
                    "wdl": wdl,
                    "game_id": f"syn_{args.seed}_{g}",
                    "ply": ply,
                    "teacher_value": value,
                    "student_value": 0.0,
                    "teacher_version": "synthetic_v01",
                    "value_perspective": "side_to_move",
                }
            )
    kept, stats = P.clean_pipeline(records)
    P.save_jsonl(args.out, kept)
    print(json.dumps({"raw": len(records), "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
