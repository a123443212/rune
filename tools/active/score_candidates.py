import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_bridge"))

import torch

from reader import load_dataset


def wdl_entropy(probs):
    import math

    return -sum(p * math.log(max(p, 1e-12)) for p in probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--pairs", default="")
    args = ap.parse_args()

    _, _, recs = load_dataset(args.shards)
    student = {}
    if args.checkpoint_dir:
        from training.trainer.trainer import Trainer

        with open(os.path.join(args.checkpoint_dir, "meta.json")) as f:
            meta = json.load(f)
        trainer = Trainer(meta["config"])
        trainer.load_checkpoint(args.checkpoint_dir)
        trainer.model.eval()
        from training.datasets.rune_dataset import make_loader

        pool = [{"fen": r["fen"], "value": 0.0, "wdl": 1} for r in recs]
        loader, _ = make_loader(pool, batch_size=256, shuffle=False, seed=0)
        vals, probs, uncs = [], [], []
        with torch.no_grad():
            for batch in loader:
                ids, masks = batch[0], batch[1]
                out = trainer.model(ids, masks)
                v, w = (out[2], out[3]) if len(out) > 2 else (out[0], out[1])
                u = out[5] if len(out) > 5 else None
                vals += v.tolist()
                probs += torch.softmax(w, dim=-1).tolist()
                uncs += u.tolist() if u is not None else [None] * len(v)
        for r, v, p, u in zip(recs, vals, probs, uncs):
            student[r["fen"]] = (v, p, u)

    rank_dis, instability = {}, {}
    if args.pairs:
        for line in open(args.pairs):
            if not line.strip():
                continue
            p = json.loads(line)
            kids = p.get("children", [])
            if len(kids) >= 2:
                tvals = [c["teacher_value"] for c in kids if "teacher_value" in c]
                svals = [c.get("student_value") for c in kids]
                if len(tvals) == len(kids) and all(s is not None for s in svals):
                    disag = sum(
                        1
                        for i in range(len(kids))
                        for j in range(i + 1, len(kids))
                        if (tvals[i] - tvals[j]) * (svals[i] - svals[j]) < 0
                    )
                    rank_dis[p["parent_fen"]] = disag / max(1, len(kids) * (len(kids) - 1) // 2)
                    instability[p["parent_fen"]] = max(tvals) - min(tvals)

    n_scored = 0
    with open(args.out, "w") as f:
        for r in recs:
            rec = {"fen": r["fen"], "identity": format(r["identity"], "016x")}
            if r["fen"] in student and r["has_teacher"]:
                sv, sp, su = student[r["fen"]]
                rec["disagreement"] = abs(r["teacher_value"] - sv)
                rec["uncertainty"] = su if su is not None else wdl_entropy(sp)
                n_scored += 1
            if r["fen"] in rank_dis:
                rec["rank_disagreement"] = rank_dis[r["fen"]]
                rec["instability"] = instability[r["fen"]]
            f.write(json.dumps(rec) + "\n")
    print(json.dumps({"records": len(recs), "scored": n_scored,
                      "note": "unscored lines carry fen only; selection counts them as missing"}, indent=2))


if __name__ == "__main__":
    main()
