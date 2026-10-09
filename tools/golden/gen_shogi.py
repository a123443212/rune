import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from training.export.export import export_model
from training.games import get as get_game
from training.models.rune_models import build_model

OUT_MODELS = os.path.join(os.path.dirname(__file__), "..", "..", "spec", "test-vectors", "models")
OUT_SHOGI = os.path.join(os.path.dirname(__file__), "..", "..", "spec", "test-vectors", "shogi")

SFENS = [
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
    "lnsgkgsn1/1r5b1/pppp1pppp/4p4/9/4P4/PPPP1PPPP/1B5R1/LNSGKGSNL b 2P 10",
    "4k4/9/9/9/9/9/9/9/4R4 w - 1",
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 2",
    "ln1gkgsnl/1r5b1/p1ppppppp/1p5P1/9/9/PPPPPPP1P/1B5R1/LNSGKGSNL b S2Pb 34",
]


def feats_to_tensors(g, feats):
    ids = []
    masks = []
    for gg in range(g.num_groups):
        idx = [i for q, i in feats if q == gg]
        if idx:
            ids.append(torch.tensor([idx]))
            masks.append(torch.ones(1, len(idx)))
        else:
            ids.append(torch.zeros(1, 1, dtype=torch.long))
            masks.append(torch.zeros(1, 1))
    return ids, masks


def build():
    os.makedirs(OUT_MODELS, exist_ok=True)
    os.makedirs(OUT_SHOGI, exist_ok=True)
    g = get_game("shogi")
    torch.manual_seed(23)
    m = build_model("RUNE-MLP", game="shogi")
    m.eval()
    export_model(m, os.path.join(OUT_MODELS, "shogi-mlp-fp32.rune"), quantization="fp32")
    vectors = []
    for sfen in SFENS:
        feats = g.extract(sfen)
        ids, masks = feats_to_tensors(g, feats)
        with torch.no_grad():
            v, w = m(ids, masks)
        vectors.append({
            "sfen": sfen,
            "features": [list(f) for f in feats],
            "phase": g.phase(sfen),
            "context": g.context(sfen),
            "value": float(v[0]),
            "wdl": [float(x) for x in w[0]],
        })
    with open(os.path.join(OUT_SHOGI, "eval.json"), "w") as f:
        json.dump({
            "version": 1,
            "game": "shogi",
            "feature_version": g.feature_version,
            "model": "../models/shogi-mlp-fp32.rune",
            "vectors": vectors,
        }, f, indent=2)
    for v in vectors:
        print(v["sfen"][:40], "feats=%d" % len(v["features"]), "value=%.6f" % v["value"])


if __name__ == "__main__":
    build()
