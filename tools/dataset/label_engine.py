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
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import chess
import chess.engine

from training.datasets import pipeline as P


def cp_to_value(cp):
    cp = max(-10000, min(10000, cp))
    return 2.0 / (1.0 + 10.0 ** (-cp / 400.0)) - 1.0


def analyse_level(engine, board, depth, nodes):
    limit = chess.engine.Limit(depth=depth)
    if nodes > 0:
        limit = chess.engine.Limit(nodes=nodes)
    info = engine.analyse(board, limit)
    score = info["score"].white()
    cp = score.score(mate_score=10000)
    out = {"cp": cp, "nodes": info.get("nodes", 0), "depth": info.get("depth", depth)}
    if info.get("wdl") is not None:
        w = info["wdl"].white()
        out["wdl_probs"] = [w.wins / 1000.0, w.draws / 1000.0, w.losses / 1000.0]
    if info.get("pv") is not None:
        out["pv_len"] = len(info["pv"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--depths", default="")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash-mb", type=int, default=16)
    ap.add_argument("--teacher-id", default="")
    ap.add_argument("--meta-out", default="")
    ap.add_argument("--limit-nodes", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--copy-to-ground-truth", action="store_true")
    args = ap.parse_args()

    levels = [int(x) for x in args.depths.split(",") if x.strip()] or [args.depth]
    primary = max(levels)
    teacher_id = args.teacher_id or f"stockfish17_d{primary}"
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    threads = 1 if args.deterministic else args.threads
    engine.configure({"Threads": threads, "Hash": args.hash_mb})
    if args.deterministic:
        try:
            engine.configure({"Clear Hash": True})
        except chess.engine.EngineError:
            pass
    try:
        engine.configure({"UCI_ShowWDL": True})
    except chess.engine.EngineError:
        pass
    ident = dict(engine.id)
    try:
        evalopt = engine.options["EvalFile"]
        evalfile = evalopt.default if hasattr(evalopt, "default") else str(evalopt)
    except KeyError:
        evalfile = ""

    pool = P.load_jsonl(args.pool)
    labeled = []
    nodes_total = 0
    t0 = time.time()
    with engine:
        for r in pool:
            if args.deterministic:
                try:
                    engine.configure({"Clear Hash": True})
                except chess.engine.EngineError:
                    pass
            board = chess.Board(r["fen"])
            stm_sign = 1.0 if board.turn == chess.WHITE else -1.0
            per_level = {}
            for depth in sorted(levels):
                lv = analyse_level(engine, board, depth, args.limit_nodes)
                v_white = cp_to_value(lv["cp"])
                lv["value_stm"] = v_white * stm_sign
                lv["wdl_stm"] = 1 if abs(lv["value_stm"]) < 0.15 else (
                    0 if lv["value_stm"] > 0 else 2)
                if "wdl_probs" in lv:
                    lv["uncertainty"] = 1.0 - max(lv["wdl_probs"])
                per_level[f"d{depth}"] = lv
                nodes_total += lv["nodes"]
            top = per_level[f"d{primary}"]
            v, wdl = top["value_stm"], top["wdl_stm"]
            rec = dict(r)
            rec["teacher_v"] = v
            rec["teacher_w"] = wdl
            rec["teacher_value"] = v
            rec["teacher_wdl"] = wdl
            rec["teacher_cp"] = top["cp"]
            if "wdl_probs" in top:
                rec["teacher_wdl_probs"] = top["wdl_probs"]
            rec["teacher_levels"] = per_level
            rec["value_perspective"] = "side_to_move"
            if "uncertainty" in top:
                rec["teacher_u"] = top["uncertainty"]
            rec["teacher_id"] = teacher_id
            rec["teacher_depth"] = primary
            rec["teacher_provenance"] = {
                "engine_name": ident.get("name", ""),
                "engine_author": ident.get("author", ""),
                "eval_file": evalfile,
                "threads": threads,
                "hash_mb": args.hash_mb,
                "depths": sorted(levels),
                "nodes_limit": args.limit_nodes,
                "deterministic": args.deterministic,
            }
            if args.copy_to_ground_truth:
                rec["value"] = v
                rec["wdl"] = wdl
            labeled.append(rec)
    dt = time.time() - t0
    P.save_jsonl(args.out, labeled)
    meta = {"teacher_id": teacher_id, "engine": args.engine, "depth": primary,
            "depths": sorted(levels), "threads": threads, "hash_mb": args.hash_mb,
            "engine_name": ident.get("name", ""), "eval_file": evalfile,
            "deterministic": args.deterministic,
            "positions": len(labeled), "seconds": dt,
            "positions_per_sec": len(labeled) / max(1e-9, dt),
            "nodes_total": nodes_total,
            "warning": "teacher supervision is not free; cost reported here, never amortized silently"}
    if args.meta_out:
        with open(args.meta_out, "w") as f:
            json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
