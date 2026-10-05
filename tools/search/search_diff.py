import argparse
import json


def load_scores(path):
    with open(path) as f:
        doc = json.load(f)
    if isinstance(doc, dict) and "scores" in doc:
        return {r["fen"]: r for r in doc["scores"]}
    return {r["fen"]: r for r in doc} if isinstance(doc, list) else doc


def load_preds_lines(path):
    try:
        with open(path) as f:
            first = f.read(2)
    except Exception:
        return None
    if first.strip().startswith("{"):
        try:
            recs = [json.loads(l) for l in open(path) if l.strip()]
            if recs and "score" in recs[0]:
                return {r["fen"]: r for r in recs}
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    ra = load_preds_lines(args.a)
    rb = load_preds_lines(args.b)
    if ra is None:
        ra = load_scores(args.a)
    if rb is None:
        rb = load_scores(args.b)
    common = sorted(set(ra) & set(rb))
    div = None
    maxd = 0.0
    for fen in common:
        d = abs(float(ra[fen]["score"]) - float(rb[fen]["score"]))
        maxd = max(maxd, d)
        if d > args.tol and div is None:
            div = {"fen": fen, "a": ra[fen]["score"], "b": rb[fen]["score"], "diff": d}
    out = {"common": len(common), "max_abs_diff": maxd, "first_divergence": div, "status": "PASS" if div is None else "DIVERGED"}
    print(json.dumps(out, indent=2))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
    return 0 if div is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
