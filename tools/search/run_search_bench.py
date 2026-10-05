import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from training.engine.cache import EvalCache
from training.engine.lazy import LazyConfig
from training.engine.scale import engine_score


class NullBoard:
    def __init__(self, fen):
        self.fen = fen

    def legal_moves(self):
        return []

    def make_move(self, m):
        return False

    def unmake_move(self):
        pass


class FileEvaluator:
    def __init__(self, preds_path, model_hash="file"):
        self.table = {}
        for l in open(preds_path):
            if not l.strip():
                continue
            r = json.loads(l)
            self.table[r["fen"]] = (float(r["pred_value"]), r.get("pred_wdl", [0.0, 1.0, 0.0]))
        self.model_hash = model_hash
        self.evals = 0
        self.lat_us = 80.0

    def evaluate(self, board):
        self.evals += 1
        fen = board.fen if isinstance(board, NullBoard) else str(board)
        if fen in self.table:
            return self.table[fen][0]
        return 0.0

    def order_moves(self, board, moves):
        return list(moves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lazy", default="L0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    fens = [l.strip() for l in open(args.positions) if l.strip()]
    ev = FileEvaluator(args.preds)
    cfg = LazyConfig(mode=args.lazy)
    from training.engine.search import AlphaBeta
    t0 = time.perf_counter()
    scores = []
    for fen in fens:
        ab = AlphaBeta(ev, cfg)
        s, m, st = ab.search(NullBoard(fen), args.depth)
        scores.append({"fen": fen, "score": s, "engine": engine_score(max(-1.0, min(1.0, s)))})
    dt = time.perf_counter() - t0
    out = {"positions": len(fens), "depth": args.depth, "lazy": args.lazy, "seconds": dt, "evals": ev.evals, "scores": scores}
    print(json.dumps({k: out[k] for k in ("positions", "depth", "lazy", "seconds", "evals")}, indent=2))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
