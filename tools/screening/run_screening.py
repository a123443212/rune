import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from training.datasets.pipeline import load_jsonl
from training.experiments.screening import ScreeningRunner, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pool", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    pool = load_jsonl(args.pool)
    out = ScreeningRunner(cfg).run(pool)
    print(json.dumps({"runs": out}, indent=2))


if __name__ == "__main__":
    main()
