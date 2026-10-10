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

import hashlib
import json

from training.features.python_features import game_phase, normalized_key, parse_fen

REQUIRED_FIELDS = ["fen", "value", "wdl"]


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def integrity_check(record, strict_legal=True):
    for field in REQUIRED_FIELDS:
        if field not in record:
            return False, f"missing {field}"
    try:
        board, stm, mask, ep_sq = parse_fen(record["fen"])
    except Exception:
        return False, "fen parse error"
    wk = sum(1 for c in board if c is not None and c == ("k", "w"))
    bk = sum(1 for c in board if c is not None and c == ("k", "b"))
    if wk != 1 or bk != 1:
        return False, "king count"
    for sq in range(64):
        cell = board[sq]
        if cell is not None and cell[0] == "p" and (sq >> 3) in (0, 7):
            return False, "pawn on back rank"
    if not (-1.0 <= record["value"] <= 1.0):
        return False, "value range"
    if record["wdl"] not in (0, 1, 2):
        return False, "wdl range"
    return True, "ok"


def filter_integrity(records):
    kept = []
    stats = {"total": len(records), "rejected": 0, "reasons": {}}
    for r in records:
        ok, reason = integrity_check(r)
        if ok:
            kept.append(r)
        else:
            stats["rejected"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
    return kept, stats


def deduplicate(records):
    seen = set()
    kept = []
    dupes = 0
    for r in records:
        key = normalized_key(r["fen"])
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        kept.append(r)
    return kept, {"duplicates": dupes, "kept": len(kept)}


def quality_filter(records, min_ply=4, max_halfmove=100):
    kept = []
    for r in records:
        if r.get("ply", 8) < min_ply:
            continue
        parts = r["fen"].split()
        try:
            if len(parts) >= 5 and int(parts[4]) > max_halfmove:
                continue
        except ValueError:
            continue
        kept.append(r)
    return kept, {"kept": len(kept)}


def phase_of_record(record):
    if "phase" in record:
        return record["phase"]
    board, _, _, _ = parse_fen(record["fen"])
    return game_phase(board)


def phase_balance(records, max_ratio=2.0):
    buckets = {0: [], 1: [], 2: []}
    for r in records:
        buckets[phase_of_record(r)].append(r)
    counts = {k: len(v) for k, v in buckets.items()}
    nonzero = [c for c in counts.values() if c > 0]
    if not nonzero:
        return records, {"counts": counts}
    target = max(nonzero)
    cap = int(target)
    smallest = min(nonzero)
    if target > smallest * max_ratio:
        cap = int(smallest * max_ratio)
    balanced = []
    for k in (0, 1, 2):
        balanced.extend(buckets[k][:cap] if len(buckets[k]) > cap else buckets[k])
    return balanced, {"counts": counts, "cap": cap, "kept": len(balanced)}


def split_by_game(records, train_frac=0.9, val_frac=0.05, seed=0):
    groups = {}
    for r in records:
        gid = r.get("game_id", normalized_key(r["fen"]))
        groups.setdefault(gid, []).append(r)
    gids = sorted(groups.keys())
    train, val, test = [], [], []
    for gid in gids:
        h = int(hashlib.sha256(f"{seed}:{gid}".encode()).hexdigest(), 16) % 10000 / 10000.0
        if h < train_frac:
            train.extend(groups[gid])
        elif h < train_frac + val_frac:
            val.extend(groups[gid])
        else:
            test.extend(groups[gid])
    for need, donor in (("val", "train"), ("test", "train")):
        target = val if need == "val" else test
        if not target and len({r.get("game_id") for r in train}) > 1:
            games_in_train = sorted({r.get("game_id", normalized_key(r["fen"])) for r in train})
            move_gid = games_in_train[-1]
            moved = [r for r in train if r.get("game_id", normalized_key(r["fen"])) == move_gid]
            train = [r for r in train if r.get("game_id", normalized_key(r["fen"])) != move_gid]
            target.extend(moved)
    return {"train": train, "val": val, "test": test}


def sample_random(records, n, seed=0):
    import random

    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    return [records[i] for i in idx[:n]]


def sample_stratified(records, n, seed=0):
    import random

    rng = random.Random(seed)
    buckets = {0: [], 1: [], 2: []}
    for r in records:
        buckets[phase_of_record(r)].append(r)
    per = n // 3
    out = []
    for k in (0, 1, 2):
        b = buckets[k][:]
        rng.shuffle(b)
        out.extend(b[:per])
    rest = n - len(out)
    pool = [r for r in records if r not in out]
    rng.shuffle(pool)
    out.extend(pool[:rest])
    rng.shuffle(out)
    return out


def sample_disagreement(records, n, disagreement_frac=0.5, seed=0):
    import random

    rng = random.Random(seed)
    scored = [r for r in records if "teacher_value" in r and "student_value" in r]
    scored.sort(key=lambda r: abs(r["teacher_value"] - r["student_value"]), reverse=True)
    n_dis = int(n * disagreement_frac)
    n_rand = n - n_dis
    picked = scored[:n_dis]
    picked_ids = {id(r) for r in picked}
    rest = [r for r in records if id(r) not in picked_ids]
    rng.shuffle(rest)
    picked = picked + rest[:n_rand]
    rng.shuffle(picked)
    return picked


def clean_pipeline(records, min_ply=4, max_ratio=2.0):
    kept, s1 = filter_integrity(records)
    kept, s2 = deduplicate(kept)
    kept, s3 = quality_filter(kept, min_ply=min_ply)
    kept, s4 = phase_balance(kept, max_ratio=max_ratio)
    return kept, {"integrity": s1, "dedup": s2, "quality": s3, "balance": s4}
