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


def wdl_value(wdl):
    s = wdl[0] + wdl[1] + wdl[2]
    if s <= 0:
        return 0.0
    return (wdl[0] - wdl[2]) / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--tol", type=float, default=0.35)
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.preds) if l.strip()]
    groups = {}
    mismatch = 0
    for r in recs:
        t = float(r["teacher_value"])
        p = float(r["pred_value"])
        pw = wdl_value(r["pred_wdl"])
        tw = wdl_value(r["teacher_wdl"])
        b = bucket(t)
        g = groups.setdefault(b, {"n": 0, "wdl_mae": 0.0, "acc": 0})
        g["n"] += 1
        g["wdl_mae"] += abs(tw - pw)
        ta = 0 if t > 0.05 else (2 if t < -0.05 else 1)
        pa = 0 if p > 0.05 else (2 if p < -0.05 else 1)
        if ta == pa:
            g["acc"] += 1
        if abs(p - pw) > args.tol:
            mismatch += 1
    out = {}
    for b, g in groups.items():
        out[b] = {"n": g["n"], "wdl_mae": g["wdl_mae"] / max(1, g["n"]), "acc": g["acc"] / max(1, g["n"])}
    out["value_wdl_mismatch"] = mismatch
    out["mismatch_rate"] = mismatch / max(1, len(recs))
    print(json.dumps(out, indent=2))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
