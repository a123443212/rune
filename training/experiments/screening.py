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
    "rune_03A": "RUNE-03-A",
    "rune_03B": "RUNE-03-B",
    "rune_03C": "RUNE-03-C",
    "rune_03D": "RUNE-03-D",
    "rune_04_cheap": "RUNE-04",
    "rune_04_always": "RUNE-04",
    "rune_04_adaptive": "RUNE-04",
    "rune_05_unc": "RUNE-05",
    "rune_05_routing": "RUNE-05",
    "rune_05_stab": "RUNE-05",
    "rune_s1": "RUNE-03-A",
    "rune_s2": "RUNE-03-A",
    "rune_s3": "RUNE-03-A",
    "rune_s4": "RUNE-03-A",
}

CORE_MODELS = ("rune_mlp",)
EXPERIMENTAL_MODELS = ("rune_attn", "rune_attn_gab", "rune_rel_s", "rune_rel_d",
                       "rune_03A", "rune_03B", "rune_03C", "rune_03D",
                       "rune_04_cheap", "rune_04_always", "rune_04_adaptive",
                       "rune_05_unc", "rune_05_routing", "rune_05_stab",
                       "rune_s1", "rune_s2", "rune_s3", "rune_s4")

ADAPTIVE_MODES = {
    "rune_04_cheap": "cheap",
    "rune_04_always": "always",
    "rune_04_adaptive": "adaptive",
    "rune_05_unc": "adaptive",
    "rune_05_routing": "adaptive",
    "rune_05_stab": "adaptive",
}

SEARCH_ROUTING = {
    "rune_05_unc": "difficulty",
    "rune_05_routing": "both",
    "rune_05_stab": "full",
}

CORE_ARCH = {"tokens": 8, "token_dim": 32, "gate": "clip", "alpha": 1.0}


def experimental_reasons(config):
    reasons = []
    for m in config.get("models", []):
        if m in EXPERIMENTAL_MODELS:
            reasons.append(f"model {m} is experimental")
    loss = config.get("loss", {})
    if loss.get("ranking", False):
        reasons.append("ranking loss is experimental (use the L1 driver on candidates)")
    if loss.get("uncertainty", False):
        reasons.append("uncertainty loss is experimental (L1 leg only, calibration-gated)")
    if loss.get("stability", False):
        reasons.append("stability loss is experimental (L2 leg only, needs child data)")
    dist = config.get("distillation", {})
    if dist.get("enabled", False):
        reasons.append(f"distillation is experimental (task={dist.get('task', 'value_wdl')}, "
                       f"weighting={dist.get('weight_mode', 'uniform')})")
    sampling = config.get("sampling", {})
    if sampling.get("mode", "none") == "disagreement_mix":
        reasons.append("disagreement sampling is experimental (stage-gated after S0)")
    arch = config.get("architecture", {})
    for key in ("tokens", "token_dim", "gate", "alpha"):
        if arch.get(key, CORE_ARCH[key]) != CORE_ARCH[key]:
            reasons.append(f"architecture.{key}={arch.get(key)} deviates from core")
    return reasons

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
        if mode == "disagreement_mix":
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
        tcfg = {
            "arch": arch,
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
                "tokens": self.cfg.get("architecture", {}).get("tokens", 8),
                "dim": self.cfg.get("architecture", {}).get("token_dim", 32),
                "gate": self.cfg.get("architecture", {}).get("gate", "clip"),
                "alpha": self.cfg.get("architecture", {}).get("alpha", 1.0),
                "dynamic_bias": model_key == "rune_rel_d",
            },
            "dense_params": {
                "variant": arch.split("-")[-1] if arch.startswith("RUNE-03-") else "A",
                "token_dims": self.cfg.get("architecture", {}).get("token_dims", [32] * 8),
                "pooling": self.cfg.get("architecture", {}).get("pooling", "none"),
                "pool_clip": self.cfg.get("architecture", {}).get("pool_clip", True),
                "gate_on": self.cfg.get("architecture", {}).get("gate_on", False),
                "shared_width": self.cfg.get("architecture", {}).get("shared_width", 32),
                "head_h1": self.cfg.get("architecture", {}).get("head_h1", 128),
                "head_h2": self.cfg.get("architecture", {}).get("head_h2", 32),
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
            "adaptive_params": {
                "mode": ADAPTIVE_MODES.get(model_key, "adaptive"),
                "search_routing": SEARCH_ROUTING.get(model_key, "difficulty"),
                "dim": self.cfg.get("architecture", {}).get("token_dim", 32),
                "cheap_pooling": self.cfg.get("architecture", {}).get("cheap_pooling", "none"),
                "alpha": self.cfg.get("architecture", {}).get("alpha", 1.0),
                "threshold": self.cfg.get("routing", {}).get("threshold", 0.5),
                "t_high": self.cfg.get("routing", {}).get("t_high", None),
                "t_low": self.cfg.get("routing", {}).get("t_low", None),
                "pruned_pairs": self.cfg.get("architecture", {}).get("pruned_pairs", []),
                "refine_precision": self.cfg.get("precision", {}).get("refine", "fp32"),
                "cheap_hidden": self.cfg.get("architecture", {}).get("cheap_hidden", 32),
                "ref_h1": self.cfg.get("architecture", {}).get("ref_h1", 128),
                "ref_h2": self.cfg.get("architecture", {}).get("ref_h2", 32),
                "lambda_diff": self.cfg.get("loss", {}).get("lambda_diff", 0.1),
                "diff_margin": self.cfg.get("loss", {}).get("diff_margin", 0.1),
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
            export_model(trainer.model, os.path.join(ckpt_dir, f"{model_key}_{tag}.rune"),
                         quantization=self.cfg.get("export_quant", "fp32"))
            record = {
                "experiment_id": self.exp_name,
                "model": model_key,
                "architecture": arch,
                "architecture_version": "0.5.0" if arch.startswith("RUNE-05") else ("0.4.0" if arch.startswith("RUNE-04") else ("0.3.0" if arch.startswith("RUNE-03-") else ("0.2.0" if arch == "RUNE-REL-02" else "0.1.0"))),
                "routing": {"mode": tcfg["adaptive_params"]["mode"],
                            "search_routing": tcfg["adaptive_params"].get("search_routing", "difficulty"),
                            **self.cfg.get("routing", {})} if arch.startswith(("RUNE-04", "RUNE-05")) else {"mode": "none"},
                "precision": self.cfg.get("precision", {"base": "fp32"}) if arch.startswith(("RUNE-04", "RUNE-05")) else {},
                "experimental": experimental_reasons(self.cfg),
                "milestone_positions": ms,
                "trained_positions": trainer.positions_seen,
                "feature_set": "grouped_hkav2_fullthreats_v01",
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
