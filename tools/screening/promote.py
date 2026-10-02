import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def load_latest(runs_dir):
    best = {}
    for path in glob.glob(os.path.join(runs_dir, "*", "*", "metrics.json")):
        with open(path) as f:
            m = json.load(f)
        key = m["model"]
        if key not in best or m["trained_positions"] > best[key]["trained_positions"]:
            best[key] = m
    return best


def trend_ok(runs_dir, model, key="value"):
    vals = []
    for path in sorted(glob.glob(os.path.join(runs_dir, model, "*", "metrics.json"))):
        with open(path) as f:
            m = json.load(f)
        vals.append((m["trained_positions"], m["val"][key]))
    vals.sort()
    if len(vals) < 2:
        return False, "need at least 2 milestones"
    first = sum(v for _, v in vals[:1]) / 1
    last = vals[-1][1]
    return last < first, f"first={first:.5f} last={last:.5f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--write-250m", default="")
    ap.add_argument("--model", default="")
    args = ap.parse_args()

    best = load_latest(args.runs)
    print("| Model | Positions | Val loss | WDL acc | Rank acc | Grad NaN | Ckpt bytes |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for model, m in sorted(best.items()):
        size = -1
        for d in sorted(glob.glob(os.path.join(args.runs, model, "*"))):
            cand = os.path.join(d, "model.pt")
            if os.path.exists(cand):
                size = os.path.getsize(cand)
        print(f"| {model} | {m['trained_positions']} | {m['val']['value']:.5f} | "
              f"{m['val']['wdl_acc']:.4f} | {m['val']['rank_acc']:.4f} | "
              f"{m['train']['grad_nan']} | {size} |")
    print()
    for model in sorted(best):
        ok, detail = trend_ok(args.runs, model)
        stable = best[model]["train"]["grad_nan"] == 0
        print(f"{model}: val-trend-down={ok} ({detail}) grad-stable={stable}")
    print()
    print("Promotion is a human decision on this evidence. No model is auto-selected.")
    if args.write_250m:
        if not args.model or args.model not in best:
            print("pass --model <id> to write an explicit 250M config")
            return
        cfg = {
            "experiment": {"name": best[args.model]["experiment_id"] + "_250m", "seed": best[args.model]["seed"]},
            "out_dir": os.path.join(args.runs + "_250m"),
            "data": {"dataset": "clean_v01", "positions": [250_000_000]},
            "models": [args.model],
            "training": {"lr": 3e-4, "batch_size": 256, "log_every": 200},
            "loss": best[args.model]["loss_config"],
            "note": "explicit promotion config, human approved",
        }
        with open(args.write_250m, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"wrote explicit 250M config for {args.model} to {args.write_250m}")


if __name__ == "__main__":
    main()
