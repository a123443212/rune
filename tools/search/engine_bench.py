import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def run(cmd):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lazy", default="L0")
    ap.add_argument("--positions", default="benchmark/positions/tactical.epd")
    ap.add_argument("--fen", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    fens = [args.fen] if args.fen else [l.strip() for l in open(args.positions) if l.strip()][:10]
    rep = {"model": args.model, "depth": args.depth, "lazy": args.lazy, "positions": []}
    for fen in fens:
        out = run(["build/rune_search", "--fen", fen, "--depth", str(args.depth), "--lazy", args.lazy])
        d = {}
        toks = []
        for line in out.splitlines():
            toks += line.strip().split()
        for i in range(0, len(toks) - 1, 2):
            try:
                d[toks[i]] = float(toks[i + 1])
            except Exception:
                pass
        d["fen"] = fen
        rep["positions"].append(d)
        print("%s score %.4f nodes %.0f nps %.0f" % (fen[:40], d.get("score", 0), d.get("nodes", 0), d.get("nps", 0)))
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
