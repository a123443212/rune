import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from training.datasets import pipeline as P
from training.datasets.flex_dataset import make_flex_loader
from training.datasets.rune_dataset import make_loader
from training.models.relational import build_rel_model
from training.models.rune_models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rel-params", default="{}")
    args = ap.parse_args()

    pool = P.load_jsonl(args.pool)
    if args.arch == "RUNE-REL-02":
        p = json.loads(args.rel_params)
        model = build_rel_model(tokens=p.get("tokens", 8), dim=p.get("dim", 32),
                                gate=p.get("gate", "clip"), alpha=p.get("alpha", 1.0),
                                dynamic_bias=p.get("dynamic_bias", False))
        loader, _ = make_flex_loader(pool, batch_size=256, shuffle=False)
    else:
        model = build_model(args.arch)
        loader, _ = make_loader(pool, batch_size=256, shuffle=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:
                ids, masks, ctx, _, _ = batch
                v, w = model(ids, masks, ctx)
            else:
                ids, masks, _, _ = batch
                v, w = model(ids, masks)
            preds = list(zip(v.tolist(), w.argmax(dim=-1).tolist()))
            for r, (sv, sw) in zip(pool[len(out):len(out) + len(preds)], preds):
                out.append({**r, "student_value": sv, "student_wdl": sw})
    P.save_jsonl(args.out, out)
    print(json.dumps({"scored": len(out)}, indent=2))


if __name__ == "__main__":
    main()
