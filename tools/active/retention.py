# RUNE — Relational Unified Neural Evaluator
# Copyright (C) 2026 a123443212
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# This project is dual-licensed under the MIT License and the
# Apache License, Version 2.0. You may choose either license
# when using, copying, modifying, or distributing this software.
#
# MIT License: https://opensource.org/license/mit
# Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, this
# software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied.

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    from training.datasets import pipeline as P
    from training.datasets.rune_dataset import make_loader
    from training.trainer.trainer import Trainer

    probe = P.load_jsonl(args.probe)
    subsets = {}
    for r in probe:
        subsets.setdefault(r.get("subset", "all"), []).append(r)

    rows = []
    for ckpt in args.checkpoints:
        with open(os.path.join(ckpt, "meta.json")) as f:
            meta = json.load(f)
        trainer = Trainer(meta["config"])
        trainer.load_checkpoint(ckpt)
        trainer.model.eval()
        row = {"checkpoint": ckpt, "arch": meta.get("arch"),
               "trained_positions": meta.get("positions_seen")}
        with torch.no_grad():
            for name, recs in subsets.items():
                loader, _ = make_loader(recs, batch_size=args.batch_size,
                                        shuffle=False, seed=0)
                se, n = 0.0, 0
                for batch in loader:
                    ids, masks = batch[0], batch[1]
                    value = batch[3]
                    out = trainer.model(ids, masks)
                    v = out[2] if len(out) > 2 else out[0]
                    se += ((v - value.to(v.device)) ** 2).sum().item()
                    n += len(value)
                row[f"mse_{name}"] = se / max(1, n)
        rows.append(row)
    with open(args.out, "w") as f:
        json.dump({"rows": rows,
                   "warning": "rising probe MSE on old subsets = forgetting, not noise"}, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
