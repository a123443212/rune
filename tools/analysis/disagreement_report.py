import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from training.datasets import pipeline as P
from training.datasets.composition import composition
from training.samplers.disagreement import disagreement_metrics, sample_mixture, score_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--ratio", type=float, default=0.2)
    ap.add_argument("--mode", default="stratified")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = P.load_jsonl(args.pool)
    scored = score_records(pool)
    metrics = disagreement_metrics(scored)
    before = composition(pool)
    picked = sample_mixture(pool, min(args.n, len(pool)), mode=args.mode,
                            disagreement_ratio=args.ratio, seed=args.seed)
    after = composition(picked)
    report = {
        "teacher_student": metrics,
        "composition_before": before,
        "composition_after": after,
        "sampling": {"mode": args.mode, "disagreement_ratio": args.ratio, "n": len(picked)},
        "warning": "disagreement saving claims must include teacher labeling cost",
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
