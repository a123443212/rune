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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch

from training.features.python_features import extract_features, game_phase, parse_fen
from training.models.dense import build_dense_model

GROUP_NAMES = [
    "pawn",
    "king",
    "minor",
    "rook",
    "queen",
    "threat",
    "mobility",
    "global",
]


def build_batch(fens):
    from training.features.python_features import VOCAB_SIZES

    per = []
    for fen in fens:
        feats = extract_features(fen)
        per.append([[i for gg, i in feats if gg == g] for g in range(9)])
    maxlen = [max(1, max(len(per[b][g]) for b in range(len(fens)))) for g in range(9)]
    batch_ids, batch_masks = [], []
    for g in range(9):
        gid = torch.zeros(len(fens), maxlen[g], dtype=torch.long)
        gm = torch.zeros(len(fens), maxlen[g], dtype=torch.float32)
        for b in range(len(fens)):
            for j, v in enumerate(per[b][g]):
                gid[b, j] = v % VOCAB_SIZES[g]
                gm[b, j] = 1.0
        batch_ids.append(gid)
        batch_masks.append(gm)
    return batch_ids, batch_masks


def collect_tokens(model, fens, batch_size=256):
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(fens), batch_size):
            chunk = fens[s:s + batch_size]
            ids, masks = build_batch(chunk)
            acc = model.embedder(ids, masks)
            segs = model.pool.split(acc, model.group_widths)
            flat = torch.cat(model.pool(segs), dim=1)
            outs.append(model.gate(flat))
    return torch.cat(outs, dim=0).numpy()


def split_widths(model):
    if model.pooling == "shared":
        return list(model.token_dims)
    return list(model.group_widths)


def token_views(flat, widths):
    views, off = [], 0
    for w in widths:
        views.append(flat[:, off:off + w])
        off += w
    return views


def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def token_token_similarity(flat, widths):
    views = token_views(flat, widths)
    n = len(views)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = cosine(views[i].reshape(-1), views[j].reshape(-1))
    return mat


def channel_correlation(flat):
    c = flat - flat.mean(axis=0, keepdims=True)
    std = c.std(axis=0, keepdims=True) + 1e-12
    corr = (c.T @ c) / (flat.shape[0] * (std.T @ std) + 1e-12)
    off = corr - np.eye(corr.shape[0])
    return float(np.abs(off).mean()), float(np.abs(off).max())


def effective_rank(flat):
    s = np.linalg.svd(flat - flat.mean(axis=0, keepdims=True), compute_uv=False)
    denom = float((s ** 2).sum())
    if denom <= 0:
        return 0.0, s.tolist()
    return float((s.sum() ** 2) / denom), s.tolist()


def activation_entropy(flat, bins=32):
    ents = []
    for c in range(flat.shape[1]):
        h, _ = np.histogram(flat[:, c], bins=bins)
        p = h.astype(float) / max(1, h.sum())
        p = p[p > 0]
        ents.append(float(-(p * np.log(p)).sum()))
    return float(np.mean(ents)), ents


def dead_channel_ratio(flat, eps=1e-6):
    var = flat.var(axis=0)
    return float((var < eps).mean()), var.tolist()


def phase_of(fen):
    board, _, _, _ = parse_fen(fen)
    return ("opening", "middlegame", "endgame")[game_phase(board)]


def summarize(flat, widths, fens):
    sim = token_token_similarity(flat, widths)
    mean_corr, max_corr = channel_correlation(flat)
    erank, spectrum = effective_rank(flat)
    ent_mean, _ = activation_entropy(flat)
    dead_ratio, var = dead_channel_ratio(flat)
    views = token_views(flat, widths)
    report = {
        "n": len(fens),
        "widths": widths,
        "total_dims": int(sum(widths)),
        "token_token_cosine": sim.tolist(),
        "token_names": GROUP_NAMES,
        "channel_mean_abs_corr": mean_corr,
        "channel_max_abs_corr": max_corr,
        "effective_rank": erank,
        "rank_ratio": erank / max(1, sum(widths)),
        "activation_entropy_mean": ent_mean,
        "dead_channel_ratio": dead_ratio,
        "channel_variance": var,
        "token_variance": [float(v.var()) for v in views],
        "token_norm_mean": [float(np.linalg.norm(v, axis=1).mean()) for v in views],
    }
    phases = {}
    for ph in ("opening", "middlegame", "endgame"):
        idx = [i for i, f in enumerate(fens) if phase_of(f) == ph]
        if len(idx) < 8:
            continue
        sub = flat[np.array(idx)]
        pr, _ = effective_rank(sub)
        dr, _ = dead_channel_ratio(sub)
        phases[ph] = {"n": len(idx), "effective_rank": pr, "dead_ratio": dr,
                      "token_variance": [float(v.var()) for v in token_views(sub, widths)]}
    report["phases"] = phases
    return report


def difficulty_proxies(flat, widths, fens):
    views = token_views(flat, widths)
    norms = np.stack([np.linalg.norm(v, axis=1) for v in views], axis=1)
    spread = norms.std(axis=1)
    sat = ((flat <= 1e-6) | (flat >= 1.0 - 1e-6)).mean(axis=1)
    per = []
    for i, f in enumerate(fens):
        per.append({"fen": f, "token_disagreement": float(spread[i]),
                    "saturation_ratio": float(sat[i])})
    order = np.argsort(-spread)
    return {
        "proxy_warning": "unsupervised proxies only, never routing labels",
        "mean_disagreement": float(spread.mean()),
        "mean_saturation": float(sat.mean()),
        "top_disagreement": [per[i] for i in order[:10].tolist()],
        "by_phase": {ph: float(spread[[j for j, f in enumerate(fens)
                                       if phase_of(f) == ph]].mean())
                      if any(phase_of(f) == ph for f in fens) else 0.0
                      for ph in ("opening", "middlegame", "endgame")},
    }


def pair_sensitivity(model, pairs):
    out = []
    model.eval()
    with torch.no_grad():
        for fa, fb in pairs:
            ta = collect_tokens(model, [fa])
            tb = collect_tokens(model, [fb])
            widths = split_widths(model)
            va, wa = model(*build_batch([fa]))
            vb, wb = model(*build_batch([fb]))
            d_tok = (tb - ta)[0]
            per = []
            off = 0
            for w in widths:
                per.append(float(np.linalg.norm(d_tok[off:off + w])))
                off += w
            out.append({
                "a": fa, "b": fb,
                "delta_token_norm": per,
                "delta_token_total": float(np.linalg.norm(d_tok)),
                "delta_value": float(abs(vb.item() - va.item())),
            })
    return out


def load_model(variant, token_dims, pooling, gate_on, checkpoint):
    model = build_dense_model(variant=variant, token_dims=token_dims,
                              pooling=pooling, gate_on=gate_on)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="A")
    ap.add_argument("--dims", default="32,32,32,32,32,32,32,32")
    ap.add_argument("--pooling", default="none")
    ap.add_argument("--gate", default="off", choices=["on", "off"])
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", default="")
    args = ap.parse_args()

    dims = [int(x) for x in args.dims.split(",")]
    model = load_model(args.variant, dims, args.pooling,
                       args.gate == "on", args.checkpoint or None)

    from training.datasets import pipeline as P

    rng = np.random.RandomState(args.seed)
    pool = P.load_jsonl(args.pool)
    idx = rng.choice(len(pool), size=min(args.n, len(pool)), replace=False)
    fens = [pool[i]["fen"] for i in idx]

    flat = collect_tokens(model, fens)
    report = summarize(flat, split_widths(model), fens)
    report["difficulty_proxies"] = difficulty_proxies(flat, split_widths(model), fens)
    report["config"] = {"variant": args.variant, "dims": dims,
                        "pooling": args.pooling, "gate": args.gate,
                        "checkpoint": args.checkpoint or "random-init",
                        "warning": "random-init numbers describe architecture bias only, never strength"}
    if args.pairs:
        with open(args.pairs) as f:
            pairs = [line.strip().split("\t") for line in f if line.strip()]
        report["pair_sensitivity"] = pair_sensitivity(model, pairs)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in
                      ("n", "widths", "total_dims", "channel_mean_abs_corr",
                       "channel_max_abs_corr", "effective_rank", "rank_ratio",
                       "activation_entropy_mean", "dead_channel_ratio",
                       "token_variance", "token_norm_mean")}, indent=2))


if __name__ == "__main__":
    main()
