import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.preds) if l.strip()]
    total = 0
    bflip = 0
    zflip = 0
    eflip = 0
    for r in recs:
        t = float(r["teacher_value"])
        p = float(r["pred_value"])
        total += 1
        ts = 1 if t >= args.beta else (-1 if t <= args.alpha else 0)
        ps = 1 if p >= args.beta else (-1 if p <= args.alpha else 0)
        if ts != ps:
            bflip += 1
        if (t > -0.05 and t < 0.05) != (p > -0.05 and p < 0.05):
            zflip += 1
        if (abs(t) > 0.9) != (abs(p) > 0.9):
            eflip += 1
    out = {"n": total, "boundary_flip": bflip, "boundary_flip_rate": bflip / max(1, total), "near_zero_flip": zflip, "near_zero_flip_rate": zflip / max(1, total), "extreme_flip": eflip, "extreme_flip_rate": eflip / max(1, total), "alpha": args.alpha, "beta": args.beta}
    print(json.dumps(out, indent=2))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
