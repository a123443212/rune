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

import chess

from training.datasets import pipeline as P

OPENINGS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/RNBQKBNR w KQkq - 0 5",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 9",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 35",
    "8/8/4k3/8/8/4K3/4P3/8 w - - 0 40",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-plies", type=int, default=120)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    recs = []
    for g in range(args.games):
        board = chess.Board(OPENINGS[g % len(OPENINGS)])
        ply = 0
        while ply < args.max_plies and not board.is_game_over():
            moves = list(board.legal_moves)
            if not moves:
                break
            recs.append({"fen": board.fen(), "value": 0.0, "wdl": 1,
                         "game_id": f"real_{args.seed}_{g}", "ply": ply})
            board.push(rng.choice(moves))
            ply += 1
    P.save_jsonl(args.out, recs)
    print(json.dumps({"positions": len(recs), "games": args.games}, indent=2))


if __name__ == "__main__":
    main()
