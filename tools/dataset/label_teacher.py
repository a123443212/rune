import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from training.datasets import pipeline as P
from training.features.python_features import VOCAB_SIZES, extract_features, parse_fen

OFFSETS = []
acc = 0
for v in VOCAB_SIZES:
    OFFSETS.append(acc)
    acc += v


def build_teacher(seed=1234):
    g = torch.Generator().manual_seed(seed)
    w1 = torch.randn(acc, 64, generator=g) * 0.05
    b1 = torch.randn(64, generator=g) * 0.01
    w2 = torch.randn(64, generator=g) * 0.2
    return w1, b1, w2


def teacher_value(feats, w1, b1, w2):
    x = torch.zeros(acc)
    for g, i in feats:
        x[OFFSETS[g] + i] = 1.0
    h = torch.clamp(x @ w1 + b1, 0.0, 1.0)
    return float(torch.tanh((h @ w2) * 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    w1, b1, w2 = build_teacher(args.seed)
    out = []
    for r in P.load_jsonl(args.pool):
        feats = extract_features(r["fen"])
        v_white = teacher_value(feats, w1, b1, w2)
        _, stm, _, _ = parse_fen(r["fen"])
        v = v_white if stm == 0 else -v_white
        wdl = 1 if abs(v) < 0.15 else (0 if v > 0 else 2)
        out.append({**r, "value": v, "wdl": wdl, "teacher_value": v, "teacher_wdl": wdl,
                    "teacher_id": "synth_mlp_v1", "student_value": 0.0, "student_wdl": 1,
                    "value_perspective": "side_to_move"})
    P.save_jsonl(args.out, out)
    print(json.dumps({"labeled": len(out), "teacher_id": "synth_mlp_v1"}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
