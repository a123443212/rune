import hashlib
import json
import os

from training.datasets import pipeline as P
from training.export.export import export_model
from training.trainer.system_stats import git_commit, hardware_info
from training.trainer.trainer import Trainer

MODEL_IDS = {
    "rune_mlp": "RUNE-MLP",
    "rune_attn": "RUNE-ATTN",
    "rune_attn_gab": "RUNE-ATTN-GAB",
    "rune_rel_s": "RUNE-REL-02",
    "rune_rel_d": "RUNE-REL-02",
}

MILESTONE_TAGS = {10_000_000: "10m", 25_000_000: "25m", 50_000_000: "50m", 100_000_000: "100m",
                  250_000_000: "250m", 500_000_000: "500m", 1_000_000_000: "1b"}


def milestone_tag(n):
    return MILESTONE_TAGS.get(n, f"{n}")


def dataset_hash(records):
    h = hashlib.sha256()
    for r in sorted(r["fen"] for r in records):
        h.update(r.encode())
    return h.hexdigest()[:16]


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

    def prepare_data(self, pool_records):
        kept, stats = P.clean_pipeline(pool_records)
        splits = P.split_by_game(kept, seed=self.seed)
        if not splits["val"] or not splits["train"]:
            raise ValueError("empty train/val split: grow the pool or change seed")
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
        return splits, info

    def run_model(self, model_key, splits, data_info, milestones):
        arch = MODEL_IDS[model_key]
        loss_cfg = self.cfg.get("loss", {})
        tcfg = {
            "arch": arch,
            "seed": self.seed,
            "lr": self.cfg.get("training", {}).get("lr", 3e-4),
            "weight_decay": self.cfg.get("training", {}).get("weight_decay", 0.01),
            "batch_size": self.cfg.get("training", {}).get("batch_size", 256),
            "lambda_wdl": 0.5 if loss_cfg.get("wdl", True) else 0.0,
            "lambda_rank": 0.1 if loss_cfg.get("ranking", False) else 0.0,
            "rank_margin": 0.05,
            "rel_params": {
                "tokens": self.cfg.get("architecture", {}).get("tokens", 8),
                "dim": self.cfg.get("architecture", {}).get("token_dim", 32),
                "gate": self.cfg.get("architecture", {}).get("gate", "clip"),
                "alpha": self.cfg.get("architecture", {}).get("alpha", 1.0),
                "dynamic_bias": model_key == "rune_rel_d",
            },
        }
        trainer = Trainer(tcfg)
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
            export_model(trainer.model, os.path.join(ckpt_dir, f"{model_key}_{tag}.rune"))
            record = {
                "experiment_id": self.exp_name,
                "model": model_key,
                "architecture": arch,
                "architecture_version": "0.1.0",
                "milestone_positions": ms,
                "trained_positions": trainer.positions_seen,
                "feature_set": "grouped_hkav2_fullthreats_v01",
                "teacher_id": self.cfg.get("data", {}).get("teacher", "unknown"),
                "loss_config": {"wdl": loss_cfg.get("wdl", True),
                                "ranking": loss_cfg.get("ranking", False)},
                "optimizer": "adamw",
                "learning_rate": tcfg["lr"],
                "batch_size": tcfg["batch_size"],
                "seed": self.seed,
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
