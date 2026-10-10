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

import hashlib
import json
import os

from training.datasets import pipeline as P
from training.export.export import export_model
from training.experiments.model_registry import (
    ADAPTIVE_MODES,
    CORE_ARCH,
    CORE_MODELS,
    EXPERIMENTAL_MODELS,
    MODEL_IDS,
    PAIR_MODELS,
    SEARCH_ROUTING,
    SWIGLU_MODELS,
    experimental_reasons,
    pair_enabled_for_model,
)
from training.trainer.system_stats import git_commit, hardware_info
from training.trainer.trainer import Trainer

MILESTONE_TAGS = {10_000_000: "10m", 25_000_000: "25m", 50_000_000: "50m", 100_000_000: "100m",
                  250_000_000: "250m", 500_000_000: "500m", 1_000_000_000: "1b"}


def milestone_tag(n):
    return MILESTONE_TAGS.get(n, f"{n}")


def dataset_hash(records):
    h = hashlib.sha256()
    for r in sorted(r["fen"] for r in records):
        h.update(r.encode())
    return h.hexdigest()[:16]


def teacher_hash(records):
    h = hashlib.sha256()
    n = 0
    for r in sorted(records, key=lambda r: r["fen"]):
        if "teacher_v" not in r:
            return None, 0
        h.update(f"{r['fen']}:{r['teacher_v']:.6f}:{r.get('teacher_id', '')}".encode())
        n += 1
    return h.hexdigest()[:16], n


def load_config(path):
    if path.endswith(".yaml") or path.endswith(".yml"):
        import yaml

        with open(path) as f:
            return yaml.safe_load(f)
    with open(path) as f:
        return json.load(f)


class ScreeningRunner:
    def __init__(self, config):
        self.cfg = config
        exp = config.get("experiment", {})
        self.exp_name = exp.get("name", "rune_screening")
        self.seed = exp.get("seed", 42)
        self.out_dir = config.get("out_dir", os.path.join("runs", self.exp_name))
        reasons = experimental_reasons(config)
        allowed = config.get("research", {}).get("allow_experimental", False)
        if reasons and not allowed:
            raise ValueError("experimental config without opt-in: " + "; ".join(reasons) +
                             " (set research.allow_experimental=true)")

    def prepare_data(self, pool_records):
        kept, stats = P.clean_pipeline(pool_records)
        splits = P.split_by_game(kept, seed=self.seed)
        if not splits["val"] or not splits["train"]:
            raise ValueError("empty train/val split: grow the pool or change seed")
        sampling = self.cfg.get("sampling", {})
        mode = sampling.get("mode", "none")
        if mode == "stratified":
            n = sampling.get("max_positions", len(splits["train"]))
            splits["train"] = P.sample_stratified(
                splits["train"], min(n, len(splits["train"])), seed=self.seed)
        elif mode == "disagreement_mix":
            from training.samplers.disagreement import sample_mixture

            n = sampling.get("max_positions", len(splits["train"]))
            splits["train"] = sample_mixture(
                splits["train"], min(n, len(splits["train"])),
                mode=sampling.get("base", "stratified"),
                disagreement_ratio=sampling.get("disagreement_ratio", 0.2),
                seed=self.seed)
        info = {
            "dataset_id": self.cfg.get("data", {}).get("dataset", "unknown"),
            "pool_raw": len(pool_records),
            "pool_clean": len(kept),
            "clean_stats": stats,
            "train_size": len(splits["train"]),
            "val_size": len(splits["val"]),
            "test_size": len(splits["test"]),
            "dataset_hash": dataset_hash(kept),
        }
        thash, tn = teacher_hash(kept)
        info["teacher_hash"] = thash
        info["teacher_labeled"] = tn
        return splits, info

    def run_model(self, model_key, splits, data_info, milestones):
        arch = MODEL_IDS[model_key]
        loss_cfg = self.cfg.get("loss", {})
        arch_cfg = self.cfg.get("architecture", {})
        tcfg = {
            "arch": arch,
            "pair": pair_enabled_for_model(self.cfg, model_key),
            "head": "value_swiglu" if model_key in SWIGLU_MODELS else arch_cfg.get("head", "value_wdl"),
            "head_buckets": self.cfg.get("head_buckets", 1),
            "head_h1": arch_cfg.get("head_h1", None),
            "head_h2": arch_cfg.get("head_h2", None),
            "game": self.cfg.get("game", "chess"),
            "seed": self.seed,
            "lr": self.cfg.get("training", {}).get("lr", 3e-4),
            "weight_decay": self.cfg.get("training", {}).get("weight_decay", 0.01),
            "batch_size": self.cfg.get("training", {}).get("batch_size", 256),
            "lambda_wdl": 0.5 if loss_cfg.get("wdl", True) else 0.0,
            "lambda_rank": 0.1 if loss_cfg.get("ranking", False) else 0.0,
            "rank_margin": 0.05,
            "lambda_unc": 0.2 if loss_cfg.get("uncertainty", False) else 0.0,
            "lambda_stab": 0.1 if loss_cfg.get("stability", False) else 0.0,
            "rel_params": {
                "tokens": arch_cfg.get("tokens", 8),
                "dim": arch_cfg.get("token_dim", 32),
                "gate": arch_cfg.get("gate", "clip"),
                "alpha": arch_cfg.get("alpha", 1.0),
                "dynamic_bias": model_key == "rune_rel_d",
            },
            "distill": {
                "enabled": self.cfg.get("distillation", {}).get("enabled", False),
                "task": self.cfg.get("distillation", {}).get("task", "value_wdl"),
                "alpha": self.cfg.get("distillation", {}).get("alpha", 0.5),
                "weight_mode": self.cfg.get("distillation", {}).get("weight_mode", "uniform"),
                "weight_lo": self.cfg.get("distillation", {}).get("weight_lo", 0.25),
                "weight_hi": self.cfg.get("distillation", {}).get("weight_hi", 1.0),
                "lambda_unc_distill": self.cfg.get("distillation", {}).get("lambda_unc_distill", 0.0),
            },
        }
        trainer = Trainer(tcfg)
        init_ckpt = self.cfg.get("training", {}).get("init_ckpt", "")
        init_positions = 0
        if init_ckpt:
            if not trainer.load_checkpoint(init_ckpt):
                raise ValueError(f"init_ckpt not loadable: {init_ckpt}")
            init_positions = trainer.positions_seen
            trainer.positions_seen = 0
        val = splits["val"]
        prev_dir = None
        for ms in milestones:
            tag = milestone_tag(ms)
            ckpt_dir = os.path.join(self.out_dir, model_key, tag)
            if os.path.exists(os.path.join(ckpt_dir, "metrics.json")):
                prev_dir = ckpt_dir
                with open(os.path.join(ckpt_dir, "metrics.json")) as f:
                    trainer.positions_seen = json.load(f)["trained_positions"]
                continue
            if prev_dir is not None:
                trainer.load_checkpoint(prev_dir)
            trainer.fit(splits["train"], val, max_positions=ms,
                        batch_size=tcfg["batch_size"],
                        log_every=self.cfg.get("training", {}).get("log_every", 200),
                        ckpt_dir=ckpt_dir, seed=self.seed)
            metrics = trainer.evaluate(val)
            stats = trainer.window_stats()
            export_model(trainer.model, os.path.join(ckpt_dir, f"{model_key}_{tag}.rune"),
                         quantization=self.cfg.get("export_quant", "fp32"))
            record = {
                "experiment_id": self.exp_name,
                "model": model_key,
                "architecture": arch,
                "architecture_version": "0.2.0" if model_key in PAIR_MODELS else ("0.5.0" if arch.startswith("RUNE-05") else ("0.4.0" if arch.startswith("RUNE-04") else ("0.3.0" if arch.startswith("RUNE-03-") else ("0.2.0" if arch == "RUNE-REL-02" else "0.1.0")))),
                "routing": {"mode": tcfg["adaptive_params"]["mode"],
                            "search_routing": tcfg["adaptive_params"].get("search_routing", "difficulty"),
                            **self.cfg.get("routing", {})} if arch.startswith(("RUNE-04", "RUNE-05")) else {"mode": "none"},
                "precision": self.cfg.get("precision", {"base": "fp32"}) if arch.startswith(("RUNE-04", "RUNE-05")) else {},
                "experimental": experimental_reasons(self.cfg),
                "milestone_positions": ms,
                "trained_positions": trainer.positions_seen,
                "init_ckpt": init_ckpt,
                "init_positions": init_positions,
                "feature_set": "grouped_hkav2_fullthreats_v02",
                "teacher_id": self.cfg.get("data", {}).get("teacher", "unknown"),
                "distillation": self.cfg.get("distillation", {"enabled": False}),
                "loss_config": {"wdl": loss_cfg.get("wdl", True),
                                "ranking": loss_cfg.get("ranking", False),
                                "uncertainty": loss_cfg.get("uncertainty", False),
                                "stability": loss_cfg.get("stability", False)},
                "optimizer": "adamw",
                "learning_rate": tcfg["lr"],
                "batch_size": tcfg["batch_size"],
                "seed": self.seed,
                "sampling": self.cfg.get("sampling", {"mode": "none"}),
                "export_quant": self.cfg.get("export_quant", "fp32"),
                "git_commit": git_commit(),
                "hardware": hardware_info(),
                "data_info": data_info,
                "train": stats,
                "val": metrics,
            }
            with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
                json.dump(record, f, indent=2)
            prev_dir = ckpt_dir
        return True

    def run(self, pool_records):
        milestones = self.cfg.get("data", {}).get("positions", [10_000_000])
        models = self.cfg.get("models", list(MODEL_IDS))
        splits, data_info = self.prepare_data(pool_records)
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "config.json"), "w") as f:
            json.dump(self.cfg, f, indent=2)
        with open(os.path.join(self.out_dir, "data_info.json"), "w") as f:
            json.dump(data_info, f, indent=2)
        for m in models:
            self.run_model(m, splits, data_info, milestones)
        return self.out_dir
