import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import chess
import chess.engine

from training.datasets import pipeline as P


def cp_to_value(cp):
    cp = max(-10000, min(10000, cp))
    return 2.0 / (1.0 + 10.0 ** (-cp / 400.0)) - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--teacher-id", default="")
    ap.add_argument("--meta-out", default="")
    ap.add_argument("--limit-nodes", type=int, default=0)
    ap.add_argument("--copy-to-ground-truth", action="store_true")
    args = ap.parse_args()

    teacher_id = args.teacher_id or f"stockfish17_d{args.depth}"
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"Threads": args.threads})
    try:
        engine.configure({"UCI_ShowWDL": True})
    except chess.engine.EngineError:
        pass
    limit = chess.engine.Limit(depth=args.depth)
    if args.limit_nodes > 0:
        limit = chess.engine.Limit(nodes=args.limit_nodes)

    pool = P.load_jsonl(args.pool)
    labeled = []
    nodes_total = 0
    t0 = time.time()
    with engine:
        for r in pool:
            board = chess.Board(r["fen"])
            info = engine.analyse(board, limit)
            score = info["score"].white()
            cp = score.score(mate_score=10000)
            v_white = cp_to_value(cp)
            stm_sign = 1.0 if board.turn == chess.WHITE else -1.0
            v = v_white * stm_sign
            wdl = 1 if abs(v) < 0.15 else (0 if v > 0 else 2)
            rec = dict(r)
            rec["teacher_v"] = v
            rec["teacher_w"] = wdl
            rec["teacher_cp"] = cp
            rec["value_perspective"] = "side_to_move"
            if info.get("wdl") is not None:
                w = info["wdl"].white()
                probs = [w.wins / 1000.0, w.draws / 1000.0, w.losses / 1000.0]
                rec["teacher_u"] = 1.0 - max(probs)
            rec["teacher_id"] = teacher_id
            rec["teacher_depth"] = args.depth
            if args.copy_to_ground_truth:
                rec["value"] = v
                rec["wdl"] = wdl
            nodes_total += info.get("nodes", 0)
            labeled.append(rec)
    dt = time.time() - t0
    P.save_jsonl(args.out, labeled)
    meta = {"teacher_id": teacher_id, "engine": args.engine, "depth": args.depth,
            "threads": args.threads, "positions": len(labeled), "seconds": dt,
            "positions_per_sec": len(labeled) / max(1e-9, dt),
            "nodes_total": nodes_total,
            "warning": "teacher supervision is not free; cost reported here, never amortized silently"}
    if args.meta_out:
        with open(args.meta_out, "w") as f:
            json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
