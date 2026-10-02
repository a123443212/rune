import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

COLUMNS = ["Model", "Positions", "Value Loss", "WDL Loss", "WDL Acc", "Rank Acc",
           "Params", "Size MB", "Pos/sec"]


def collect(runs_dir):
    rows = {}
    for path in glob.glob(os.path.join(runs_dir, "*", "*", "metrics.json")):
        with open(path) as f:
            m = json.load(f)
        tag = os.path.basename(os.path.dirname(path))
        rows.setdefault(tag, []).append(m)
    return rows


def row_for(m):
    val = m["val"]
    return [m["model"], m["trained_positions"], round(val["value"], 5), round(val["wdl"], 5),
            round(val["wdl_acc"], 4), round(val["rank_acc"], 4), val["params"],
            round(val["params"] * 4 / 1e6, 3), round(m["train"]["positions_per_sec"], 1)]


def markdown_table(rows):
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "|".join([" --- "] * len(COLUMNS)) + "|"]
    for m in sorted(rows, key=lambda r: r["model"]):
        lines.append("| " + " | ".join(str(c) for c in row_for(m)) + " |")
    return "\n".join(lines)


def grad_section(rows):
    lines = ["", "## Training stability", "",
             "| Model | Grad mean | Grad max | NaN steps | Elapsed s |", "| --- | --- | --- | --- | --- |"]
    for m in sorted(rows, key=lambda r: r["model"]):
        t = m["train"]
        lines.append(f"| {m['model']} | {t['grad_norm_mean']:.4f} | {t['grad_norm_max']:.4f} "
                     f"| {t['grad_nan']} | {t['elapsed_sec']:.1f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rows = collect(args.runs)
    os.makedirs(args.out_dir, exist_ok=True)
    for tag, ms in sorted(rows.items()):
        body = [f"# Screening comparison ({tag})", "",
                "No composite score is computed. Compare each metric separately under",
                "identical data, teacher, recipe, and benchmark conditions.", "",
                markdown_table(ms), grad_section(ms), ""]
        with open(os.path.join(args.out_dir, f"comparison_{tag}.md"), "w") as f:
            f.write("\n".join(body))
        print(f"wrote comparison_{tag}.md")


if __name__ == "__main__":
    main()
