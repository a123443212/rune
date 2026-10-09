import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from training.export.export import export_model
from training.features import python_features as pf
from training.models.rune_models import RuneFullModel
from tools.golden.gen_bucket_golden import batch_single

OUT_MODELS = os.path.join(os.path.dirname(__file__), "..", "..", "spec", "test-vectors", "models")
OUT_V10 = os.path.join(os.path.dirname(__file__), "..", "..", "spec", "test-vectors", "v10")

SEED = 37
MIXER_KEYS = ["wq", "bq", "wk", "bk", "wv", "bv", "gab"]


def main():
    os.makedirs(OUT_MODELS, exist_ok=True)
    os.makedirs(OUT_V10, exist_ok=True)
    torch.manual_seed(SEED)
    model = RuneFullModel("RUNE-ATTN-MH4", buckets=1)
    model.eval()
    path = os.path.join(OUT_MODELS, "small-mh4-fp32.rune")
    header = export_model(model, path, quantization="fp32")
    assert header["head_buckets"] == 1
    assert header["attention"] == "multi_head"
    names = [t["name"] for t in header["tensors"]]
    for h in range(4):
        for k in MIXER_KEYS:
            assert f"{k}_h{h}" in names, f"{k}_h{h}"
    assert "wo" in names and "bwo" in names
    assert "w1" in names and "wq" not in names

    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ]
    vectors = []
    with torch.no_grad():
        for fen in fens:
            feats = pf.extract_features(fen)
            ids, masks = batch_single(feats)
            value, wdl = model(ids, masks)
            vectors.append({
                "fen": fen,
                "value": float(value.item()),
                "wdl": [float(x) for x in wdl[0].tolist()],
            })
    doc = {
        "feature_version": pf.FEATURE_VERSION,
        "model": "small-mh4-fp32.rune",
        "seed": SEED,
        "vectors": vectors,
    }
    with open(os.path.join(OUT_V10, "multi_head.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote multi_head.json with %d vectors" % len(vectors))


if __name__ == "__main__":
    main()
