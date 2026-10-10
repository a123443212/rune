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

VALID = "VALID"
PROVISIONAL = "PROVISIONAL"
UNSTABLE = "UNSTABLE"
INVALID = "INVALID"


def assign_validity(record, rules):
    if not record.get("has_teacher", record.get("teacher_v") is not None):
        return INVALID, "no_teacher"
    v = record.get("teacher_v", 0.0)
    if not (-1.0 <= v <= 1.0):
        return INVALID, "value_range"
    if record.get("teacher_w") not in (0, 1, 2, None) and "teacher_w" in record:
        return INVALID, "wdl_range"
    if rules.get("require_provenance", False) and not record.get("teacher_provenance"):
        return PROVISIONAL, "no_provenance"
    stab = record.get("teacher_stability", None)
    unstable_at = rules.get("unstable_above", None)
    if stab is not None and unstable_at is not None and stab > unstable_at:
        return UNSTABLE, "stability"
    depth = record.get("teacher_depth", None)
    min_depth = rules.get("min_depth", None)
    if depth is not None and min_depth is not None and depth < min_depth:
        return PROVISIONAL, "shallow"
    return VALID, "ok"


def stability_from_levels(levels):
    vals = [levels[k]["value_stm"] for k in sorted(levels) if "value_stm" in levels[k]]
    if len(vals) < 2:
        return None, None
    import statistics

    diffs = [abs(b - a) for a, b in zip(vals, vals[1:])]
    flips = 0
    for a, b in zip(vals, vals[1:]):
        ca = 1 if abs(a) < 0.15 else (0 if a > 0 else 2)
        cb = 1 if abs(b) < 0.15 else (0 if b > 0 else 2)
        flips += ca != cb
    return max(diffs), {"max_delta": max(diffs), "mean_delta": statistics.mean(diffs),
                        "wdl_flips": flips, "levels": len(vals)}


def quality_weight(record, mode="uniform", lo=0.25, hi=1.0):
    if mode == "uniform":
        return 1.0
    q = record.get("target_quality", None)
    if q is None:
        return 1.0
    q = max(0.0, min(1.0, q))
    if mode == "quality":
        w = q
    elif mode == "difficulty":
        w = 1.0 - q
    else:
        raise ValueError(f"unknown quality weighting {mode}")
    return lo + (hi - lo) * w


def soft_wdl_probs(record):
    probs = record.get("teacher_wdl_probs", None)
    if probs is not None and len(probs) == 3:
        s = sum(probs)
        if s > 0:
            return [p / s for p in probs]
    w = record.get("teacher_w", record.get("teacher_wdl", 1))
    out = [0.0, 0.0, 0.0]
    out[w] = 1.0
    return out


def build_rank_pairs(scored_children, tie_margin=0.05):
    strict, soft, ties = [], [], []
    kids = sorted(scored_children, key=lambda c: -c[1])
    for i in range(len(kids)):
        for j in range(i + 1, len(kids)):
            d = kids[i][1] - kids[j][1]
            if d < tie_margin:
                ties.append((kids[i][0], kids[j][0]))
            elif d < 2 * tie_margin:
                soft.append((kids[i][0], kids[j][0], d))
            else:
                strict.append((kids[i][0], kids[j][0]))
    return {"strict": strict, "soft": soft, "ties": ties}
