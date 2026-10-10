import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch

from training.export.export import export_model
from training.games.go_v02 import GoGameV02
from training.models.resnet import build_resnet

OUT_MODELS = os.path.join(os.path.dirname(__file__), "..", "..", "spec", "test-vectors", "models")
OUT_RESNET = os.path.join(os.path.dirname(__file__), "..", "..", "spec", "test-vectors", "resnet")

SEED = 23


def empty_state(n=9):
    return "/".join(["." * n] * n) + " b - 7.5 1 0"


def single_state(n=9):
    rows = ["." * n for _ in range(n)]
    rows[n // 2] = "." * (n // 2) + "X" + "." * (n - n // 2 - 1)
    return "/".join(rows) + " b - 7.5 10 0"


def ko_state():
    rows = ["." * 9 for _ in range(9)]
    rows[4] = "....X...."
    return "/".join(rows) + " w 40 7.5 60 0"


def late_state(n=9):
    return "/".join(["." * n] * n) + " b - 7.5 70 0"


def build_for_size(n, blocks, states, name):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    m = build_resnet(board=n, channels=8, blocks=blocks, in_planes=8, feature_set="go_planes_v02")
    m.eval()
    export_model(m, os.path.join(OUT_MODELS, f"resnet{n}-8plane-fp32.rune"), quantization="fp32")
    g = GoGameV02(size=n)
    vectors = []
    for s in states:
        planes = g.planes(s)
        ctx = g.context(s)
        feats = g.extract(s)
        legal = g.legal(s)
        with torch.no_grad():
            v, w, logits, probs = m(torch.tensor(planes))
        top3 = sorted([float(x) for x in probs[0]], reverse=True)[:3]
        vectors.append({
            "state": s,
            "feature_version": g.feature_version,
            "planes_shape": list(planes.shape),
            "planes_empty_sum": float(planes[:, 2, :, :].sum()),
            "context": [float(x) for x in ctx],
            "phase": int(g.phase(s)),
            "features": [[int(a), int(b)] for a, b in feats],
            "legal_count": int(len(legal)),
            "legal_has_pass": bool(-1 in legal),
            "score_empty": None,
            "value": float(v[0]),
            "wdl": [float(x) for x in w[0]],
            "policy_top3": [float(x) for x in top3],
        })
    vectors[0]["score_empty"] = float(g.score(states[0]))
    with open(os.path.join(OUT_RESNET, name), "w") as f:
        json.dump({"version": 1, "arch": "RUNE-RESNET-01", "feature_version": "go_planes_v02", "in_planes": 8, "board": n, "vectors": vectors}, f, indent=1)
    print(f"wrote {name} with", len(vectors), "vectors")


def build():
    os.makedirs(OUT_MODELS, exist_ok=True)
    os.makedirs(OUT_RESNET, exist_ok=True)
    build_for_size(9, 2, [empty_state(9), single_state(9), ko_state(), late_state(9)], "eval_v02.json")
    build_for_size(13, 1, [empty_state(13), single_state(13), late_state(13)], "eval_v02_13.json")
    build_for_size(19, 1, [empty_state(19), single_state(19), late_state(19)], "eval_v02_19.json")


if __name__ == "__main__":
    build()
