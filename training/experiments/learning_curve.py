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

import csv
import json
import os

from training.datasets import pipeline as P
from training.datasets.rune_dataset import make_loader
from training.export.export import export_model
from training.export.quantize import quantization_report
from training.models.rune_models import build_model
from training.trainer.trainer import Trainer

MILESTONES = [100_000_000, 250_000_000, 500_000_000, 1_000_000_000]


class LearningCurve:
    def __init__(self, config):
        self.cfg = config

    def run(self, pool_records, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(self.cfg, f, indent=2)
        milestones = self.cfg.get("milestones", MILESTONES)
        sampling = self.cfg.get("sampling", "filtered")
        budgets = self.cfg.get("milestone_budgets", {})
        summary = []
        for ms in milestones:
            budget = budgets.get(str(ms), ms)
            if "max_positions_cap" in self.cfg:
                budget = min(budget, self.cfg["max_positions_cap"])
            recs = self.select(pool_records, ms, sampling)
            trainer = Trainer(self.cfg)
            val = recs["val"]
            history = trainer.fit(recs["train"], val, max_positions=budget,
                                  batch_size=self.cfg.get("batch_size", 256),
                                  log_every=self.cfg.get("log_every", 50),
                                  ckpt_dir=os.path.join(out_dir, f"ckpt_{ms}"),
                                  seed=self.cfg.get("seed", 0))
            metrics = trainer.evaluate(val)
            export_model(trainer.model, os.path.join(out_dir, f"model_{ms}.rune"),
                         quantization=self.cfg.get("export_quant", "fp32"))
            _, arrays = self._arrays(trainer.model)
            qrep = quantization_report(arrays)
            row = {
                "milestone_positions": ms,
                "trained_positions": trainer.positions_seen,
                "train_pool": len(recs["train"]),
                "val_loss": metrics["total"],
                "val_value_loss": metrics["value"],
                "wdl_acc": metrics["wdl_acc"],
                "rank_acc": metrics["rank_acc"],
                "params": metrics["params"],
                "model_size_mb": metrics["params"] * 4 / 1e6,
                "max_emb_err": max(v["max_abs_err"] for v in qrep.values()),
            }
            summary.append(row)
        with open(os.path.join(out_dir, "learning_curve.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        return summary

    def select(self, pool_records, milestone, sampling):
        ratio = self.cfg.get("pool_ratio", 1.0)
        min_pool = self.cfg.get("min_pool", 100)
        splits = P.split_by_game(pool_records, seed=self.cfg.get("seed", 0))
        train = splits["train"]
        n = min(len(train), max(min_pool, int(milestone * ratio)))
        if sampling == "random":
            train = P.sample_random(train, n, seed=self.cfg.get("seed", 0))
        elif sampling == "stratified":
            train = P.sample_stratified(train, n, seed=self.cfg.get("seed", 0))
        elif sampling == "disagreement":
            train = P.sample_disagreement(train, n, seed=self.cfg.get("seed", 0))
        return {"train": train, "val": splits["val"][:2000], "test": splits["test"][:2000]}

    @staticmethod
    def _arrays(model):
        from training.export.export import collect_tensors

        order, arrays = collect_tensors(model)
        return order, arrays
