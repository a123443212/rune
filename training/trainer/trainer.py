import json
import os
import time

import torch

from training.datasets.rune_dataset import make_loader
from training.losses.composite import CompositeLoss
from training.losses.losses import ranking_accuracy, wdl_accuracy
from training.models.relational import build_rel_model
from training.models.rune_models import build_model
from training.trainer.system_stats import file_size_bytes, git_commit, hardware_info


def derive_seed(config):
    import hashlib
    import json

    key = json.dumps({"seed": config.get("seed", 0), "arch": config.get("arch"),
                      "rel": config.get("rel_params", {})}, sort_keys=True)
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2 ** 31)


def count_ranking_pairs(values, group_size=4):
    n = (len(values) // group_size) * group_size
    if n < 2:
        return None
    a_idx, b_idx, signs = [], [], []
    for start in range(0, n, group_size):
        for j in range(group_size):
            for k in range(j + 1, group_size):
                i1, i2 = start + j, start + k
                s = 1.0 if values[i1] >= values[i2] else -1.0
                a_idx.append(i1)
                b_idx.append(i2)
                signs.append(s)
    if not a_idx:
        return None
    return (torch.tensor(a_idx), torch.tensor(b_idx), torch.tensor(signs))


class Trainer:
    def __init__(self, config):
        self.cfg = config
        torch.manual_seed(derive_seed(config))
        if config["arch"] == "RUNE-REL-02":
            p = config.get("rel_params", {})
            self.model = build_rel_model(tokens=p.get("tokens", 8), dim=p.get("dim", 32),
                                         gate=p.get("gate", "clip"), alpha=p.get("alpha", 1.0),
                                         dynamic_bias=p.get("dynamic_bias", False))
            self.needs_context = True
        elif config["arch"].startswith("RUNE-03-"):
            from training.models.dense import build_dense_model

            p = config.get("dense_params", {})
            self.model = build_dense_model(variant=config["arch"].split("-")[-1],
                                           token_dims=p.get("token_dims", [32] * 8),
                                           pooling=p.get("pooling", "none"),
                                           pool_clip=p.get("pool_clip", True),
                                           gate_on=p.get("gate_on", False),
                                           shared_width=p.get("shared_width", 32))
            self.needs_context = False
        else:
            self.model = build_model(config["arch"])
            self.needs_context = False
        self.loss_fn = CompositeLoss(
            value=True,
            wdl=config.get("lambda_wdl", 0.5) > 0,
            ranking=config.get("lambda_rank", 0.0) > 0,
            lambda_wdl=config.get("lambda_wdl", 0.5),
            lambda_rank=config.get("lambda_rank", 0.1),
            rank_margin=config.get("rank_margin", 0.05),
        )
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("lr", 3e-4),
            weight_decay=config.get("weight_decay", 0.01),
        )
        self.positions_seen = 0
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.train_start = time.time()
        self.grad_window = []
        self.step_times = []

    def unpack(self, batch):
        if len(batch) == 5:
            ids, masks, ctx, value, wdl = batch
            return ids, masks, ctx, value, wdl
        ids, masks, value, wdl = batch
        return ids, masks, None, value, wdl

    def forward_model(self, ids, masks, ctx):
        ids = [t.to(self.device) for t in ids]
        masks = [t.to(self.device) for t in masks]
        if self.needs_context:
            return self.model(ids, masks, ctx.to(self.device))
        return self.model(ids, masks)

    def train_step(self, batch, rank_batch=None):
        t0 = time.time()
        ids, masks, ctx, value, wdl = self.unpack(batch)
        value = value.to(self.device)
        wdl = wdl.to(self.device)
        self.model.train()
        v_pred, w_pred = self.forward_model(ids, masks, ctx)
        rank = None
        if rank_batch is not None:
            rids, rmasks, rctx, a_idx, b_idx, signs, weights = rank_batch
            rv_pred, _ = self.forward_model(rids, rmasks, rctx)
            a = torch.tensor(a_idx, device=self.device)
            b = torch.tensor(b_idx, device=self.device)
            rank = (rv_pred[a], rv_pred[b], torch.tensor(signs, device=self.device),
                    torch.tensor(weights, device=self.device))
        elif self.cfg.get("lambda_rank", 0.0) > 0 and len(value) >= 4:
            idx = count_ranking_pairs(value.tolist(), self.cfg.get("rank_group", 4))
            if idx is not None:
                a, b, s = idx
                rank = (v_pred[a], v_pred[b], s.to(self.device))
        losses = self.loss_fn(v_pred, w_pred, value, wdl, rank)
        self.opt.zero_grad()
        losses["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.positions_seen += len(value)
        self.grad_window.append(float(grad_norm))
        self.step_times.append(time.time() - t0)
        del self.grad_window[:-1000]
        del self.step_times[:-1000]
        return {k: v.item() for k, v in losses.items()}

    def window_stats(self):
        import math

        grads = [g for g in self.grad_window if math.isfinite(g)]
        elapsed = time.time() - self.train_start
        return {
            "grad_norm_mean": sum(grads) / max(1, len(grads)),
            "grad_norm_max": max(grads) if grads else 0.0,
            "grad_nan": len(self.grad_window) - len(grads),
            "positions_per_sec": self.positions_seen / max(1e-6, elapsed),
            "elapsed_sec": elapsed,
        }

    @torch.no_grad()
    def evaluate(self, records, batch_size=512):
        loader, _ = self.make_eval_loader(records, batch_size)
        self.model.eval()
        tot = {"total": 0.0, "value": 0.0, "wdl": 0.0, "rank": 0.0}
        racc, wacc, n = 0.0, 0.0, 0
        nb = 0
        for batch in loader:
            ids, masks, ctx, value, wdl = self.unpack(batch)
            v_pred, w_pred = self.forward_model(ids, masks, ctx)
            losses = self.loss_fn(v_pred, w_pred, value.to(self.device), wdl.to(self.device))
            for k in tot:
                tot[k] += losses[k].item()
            wacc += wdl_accuracy(w_pred, wdl).item() * len(value)
            if len(value) >= 4:
                idx = count_ranking_pairs(value.tolist())
                if idx is not None:
                    a, b, s = idx
                    racc += ranking_accuracy(v_pred[a], v_pred[b], s).item() * len(a)
                    n += len(a)
            nb += 1
        out = {k: v / max(1, nb) for k, v in tot.items()}
        out["wdl_acc"] = wacc / max(1, sum(1 for _ in records))
        out["rank_acc"] = racc / max(1, n)
        out["positions_seen"] = self.positions_seen
        out["params"] = self.model.parameter_count()
        return out

    def make_train_loader(self, records, batch_size, shuffle, seed):
        if self.needs_context:
            from training.datasets.flex_dataset import make_flex_loader

            return make_flex_loader(records, batch_size=batch_size, shuffle=shuffle, seed=seed)
        return make_loader(records, batch_size=batch_size, shuffle=shuffle, seed=seed)

    def make_eval_loader(self, records, batch_size):
        return self.make_train_loader(records, batch_size, False, 0)

    def fit(self, train_records, val_records, max_positions, batch_size=256, log_every=50,
            ckpt_dir=None, seed=0, rank_batches=None):
        loader, _ = self.make_train_loader(train_records, batch_size, True, seed)
        history = []
        it = iter(loader)
        step = 0
        while self.positions_seen < max_positions:
            try:
                batch = next(it)
            except StopIteration:
                loader, _ = self.make_train_loader(train_records, batch_size, True, seed + step)
                it = iter(loader)
                batch = next(it)
            rb = rank_batches[step % len(rank_batches)] if rank_batches else None
            losses = self.train_step(batch, rank_batch=rb)
            step += 1
            if step % log_every == 0:
                val = self.evaluate(val_records)
                row = {"step": step, **losses, **{f"val_{k}": v for k, v in val.items()}}
                history.append(row)
        if ckpt_dir:
            self.save_checkpoint(ckpt_dir)
        return history

    def load_checkpoint(self, ckpt_dir):
        model_path = os.path.join(ckpt_dir, "model.pt")
        meta_path = os.path.join(ckpt_dir, "meta.json")
        if not (os.path.exists(model_path) and os.path.exists(meta_path)):
            return False
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        with open(meta_path) as f:
            meta = json.load(f)
        self.positions_seen = meta.get("positions_seen", 0)
        return True

    def save_checkpoint(self, ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        t0 = time.time()
        model_path = os.path.join(ckpt_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)
        meta = {
            "arch": self.cfg["arch"],
            "seed": self.cfg.get("seed", 0),
            "positions_seen": self.positions_seen,
            "params": self.model.parameter_count(),
            "model_size_mb": self.model.model_size_bytes() / 1e6,
            "checkpoint_size_bytes": file_size_bytes(model_path),
            "checkpoint_sec": time.time() - t0,
            "config": self.cfg,
            "git_commit": git_commit(),
            "hardware": hardware_info(),
            "timestamp": time.time(),
        }
        with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
