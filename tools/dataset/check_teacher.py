import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from training.datasets import pipeline as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import random

    import chess
    import chess.engine

    pool = P.load_jsonl(args.pool)
    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n, len(pool)))
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"Threads": 1, "Hash": 16})
    try:
        engine.configure({"UCI_ShowWDL": True})
    except chess.engine.EngineError:
        pass
    limit = chess.engine.Limit(depth=args.depth)
    diffs, flips, max_delta = [], 0, 0.0
    with engine:
        for r in sample:
            vals = []
            for _ in range(2):
                try:
                    engine.configure({"Clear Hash": True})
                except chess.engine.EngineError:
                    pass
                board = chess.Board(r["fen"])
                info = engine.analyse(board, limit)
                cp = info["score"].white().score(mate_score=10000)
                vals.append(cp)
            d = abs(vals[0] - vals[1])
            diffs.append(d)
            max_delta = max(max_delta, d)
            w0 = 1 if abs(vals[0]) < 60 else (0 if vals[0] > 0 else 2)
            w1 = 1 if abs(vals[1]) < 60 else (0 if vals[1] > 0 else 2)
            flips += w0 != w1
    import statistics

    report = {
        "n": len(sample), "depth": args.depth,
        "exact_cp_match_rate": sum(d == 0 for d in diffs) / max(1, len(diffs)),
        "mean_abs_cp_delta": statistics.mean(diffs) if diffs else 0.0,
        "max_abs_cp_delta": max_delta,
        "wdl_flip_rate": flips / max(1, len(sample)),
        "verdict": "inspect any flip/delta before blaming training noise",
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
