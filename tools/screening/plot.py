import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect_series(runs_dir):
    series = {}
    for path in glob.glob(os.path.join(runs_dir, "*", "*", "metrics.json")):
        with open(path) as f:
            m = json.load(f)
        series.setdefault(m["model"], []).append((m["trained_positions"], m))
    for v in series.values():
        v.sort()
    return series


def curve(out, title, ylabel, get, series, logx=True):
    plt.figure()
    for model, pts in sorted(series.items()):
        xs = [p[0] for p in pts]
        ys = [get(p[1]) for p in pts]
        plt.plot(xs, ys, marker="o", label=model)
    plt.xlabel("training positions")
    plt.ylabel(ylabel)
    plt.title(title)
    if logx:
        plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    series = collect_series(args.runs)
    os.makedirs(args.out_dir, exist_ok=True)
    curve(os.path.join(args.out_dir, "val_loss.png"), "Validation loss vs positions",
          "value loss", lambda m: m["val"]["value"], series)
    curve(os.path.join(args.out_dir, "wdl_acc.png"), "WDL accuracy vs positions",
          "accuracy", lambda m: m["val"]["wdl_acc"], series)
    curve(os.path.join(args.out_dir, "rank_acc.png"), "Ranking accuracy vs positions",
          "accuracy", lambda m: m["val"]["rank_acc"], series)
    curve(os.path.join(args.out_dir, "throughput.png"), "Training throughput vs positions",
          "positions/sec", lambda m: m["train"]["positions_per_sec"], series)
    print("wrote 4 plots to", args.out_dir)


if __name__ == "__main__":
    main()
