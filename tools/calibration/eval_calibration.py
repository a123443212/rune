import argparse
import json
import math


def bucket(v):
    a = abs(float(v))
    if a < 0.1:
        return "near-equal"
    if a < 0.35:
        return "medium"
    if a < 0.7:
        return "high"
    return "winning-losing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.preds) if l.strip()]
    groups = {}
    for r in recs:
        t = float(r["teacher_value"])
        p = float(r["pred_value"])
        b = bucket(t)
        g = groups.setdefault(b, {"n": 0, "ae": 0.0, "se": 0.0, "sign_err": 0})
        g["n"] += 1
        g["ae"] += abs(t - p)
        g["se"] += (t - p) ** 2
        if (t > 0.05 and p < -0.05) or (t < -0.05 and p > 0.05):
            g["sign_err"] += 1
    out = {}
    for b, g in groups.items():
        out[b] = {"n": g["n"], "mae": g["ae"] / max(1, g["n"]), "rmse": math.sqrt(g["se"] / max(1, g["n"])), "sign_err_rate": g["sign_err"] / max(1, g["n"])}
    print(json.dumps(out, indent=2))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
