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
import copy
import json
import os
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from training.datasets import pipeline as P


def run(cmd, cwd):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--rune-data", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..", "..")
    with open(args.config) as f:
        cfg = json.load(f)
    os.makedirs(args.state_dir, exist_ok=True)
    state_path = os.path.join(args.state_dir, "state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"rounds": []}
    start = len(state["rounds"])

    for r in range(start, start + args.rounds):
        rd = os.path.join(args.state_dir, f"active-round-{r:03d}")
        os.makedirs(rd, exist_ok=True)
        prev_ckpt = state["rounds"][-1]["checkpoint"] if state["rounds"] else ""
        prev_pool = state["rounds"][-1]["pool"] if state["rounds"] else cfg["seed_pool"]
        score_ckpt = prev_ckpt or cfg.get("seed_checkpoint", "")
        if not score_ckpt:
            raise ValueError("cold start needs seed_checkpoint (see §17): train a seed model first")

        scored = os.path.join(rd, "scores.jsonl")
        run([sys.executable, "tools/active/score_candidates.py", "--shards",
             cfg["candidate_shards"], "--out", scored, "--checkpoint-dir", score_ckpt,
             ] + (["--pairs", cfg["pairs"]] if cfg.get("pairs") else []), root)
        sel = os.path.join(rd, "selected")
        weights = cfg.get("weights", {"disagreement": 1.0, "uncertainty": 0.0,
                                      "instability": 0.0, "rarity": 0.0})
        run([args.rune_data, "select", "--input", cfg["candidate_shards"], "--out", sel,
             "--dataset-id", cfg["dataset_id"], "--seed", str(cfg["seed"]),
             "--budget", str(cfg["budget"]), "--method", cfg["method"], "--round", str(r),
             "--w-disagreement", str(weights.get("disagreement", 0.0)),
             "--w-uncertainty", str(weights.get("uncertainty", 0.0)),
             "--w-instability", str(weights.get("instability", 0.0)),
             "--w-rarity", str(weights.get("rarity", 0.0)),
             "--diversity-cap", str(cfg.get("diversity_cap", 1000000)),
             "--floor-ratio", str(cfg.get("floor_ratio", 0.2)),
             "--teacher-version", cfg["teacher_id"],
             "--student-version", prev_ckpt or "seed"], root)
        audit = [json.loads(l) for l in open(os.path.join(sel, "selection_audit.jsonl"))
                 if l.strip()]
        lab_pool = os.path.join(rd, "requested.jsonl")
        with open(lab_pool, "w") as f:
            for i, a in enumerate(audit):
                gid = f"round{r}_{a.get('game_hash', str(i))}"[:64]
                f.write(json.dumps({"fen": a["fen"], "value": 0.0, "wdl": 1,
                                    "game_id": gid, "ply": 8}) + "\n")
        labeled = os.path.join(rd, "labeled.jsonl")
        run([sys.executable, "tools/dataset/label_engine.py", "--pool", lab_pool,
             "--out", labeled, "--engine", cfg["teacher_engine"],
             "--depth", str(cfg.get("teacher_depth", 10)),
             "--teacher-id", cfg["teacher_id"], "--copy-to-ground-truth"], root)
        round_pool = os.path.join(rd, "pool.jsonl")
        if cfg.get("train_mode", "mixed") == "mixed" and state["rounds"]:
            base = P.load_jsonl(state["rounds"][-1]["pool"])
        else:
            base = []
        if cfg.get("replay_ratio", 0.0) > 0 and state["rounds"]:
            import random

            rng = random.Random(cfg["seed"] + r)
            keep = int(len(base) * cfg["replay_ratio"])
            base = rng.sample(base, min(keep, len(base)))
        new_labels = P.load_jsonl(labeled)
        with open(round_pool, "w") as f:
            for rec in base + new_labels:
                f.write(json.dumps(rec) + "\n")
        train_cfg = copy.deepcopy(yaml.safe_load(open(cfg["train_config"])))
        train_cfg["experiment"]["name"] = f"{cfg['dataset_id']}_r{r}"
        train_cfg["out_dir"] = os.path.join(rd, "train")
        train_cfg["data"]["dataset"] = f"{cfg['dataset_id']}_r{r}"
        train_cfg["data"]["teacher"] = cfg["teacher_id"]
        if cfg.get("train_mode", "mixed") == "continue" and prev_ckpt:
            train_cfg["training"]["init_ckpt"] = prev_ckpt
        train_cfg_path = os.path.join(rd, "train.yaml")
        with open(train_cfg_path, "w") as f:
            yaml.safe_dump(train_cfg, f)
        run([sys.executable, "tools/screening/run_screening.py", "--config",
             train_cfg_path, "--pool", round_pool], root)
        ckpt = os.path.join(rd, "train", cfg["train_model_key"],
                            cfg["train_milestone_tag"])
        cov = os.path.join(rd, "coverage.json")
        run([sys.executable, "tools/active/coverage.py", "--pool", cfg["candidate_shards"],
             "--dataset", round_pool, "--batch",
             os.path.join(sel, "shard-00000"), "--out", cov], root)
        entry = {"round": r, "checkpoint": ckpt, "pool": round_pool,
                 "selection": sel, "coverage": cov,
                 "labels_requested": len(audit), "train_config": train_cfg_path}
        state["rounds"].append(entry)
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        print(json.dumps({"round": r, "checkpoint": ckpt,
                          "labels_requested": len(audit)}, indent=2))


if __name__ == "__main__":
    main()
