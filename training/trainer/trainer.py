import json
import os
import time

import torch

from training.datasets.rune_dataset import make_loader
from training.losses.losses import RuneLoss, ranking_accuracy, wdl_accuracy
from training.models.rune_models import build_model


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
        torch.manual_seed(config.get("seed", 0))
        self.model = build_model(config["arch"])
        self.loss_fn = RuneLoss(
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

    def train_step(self, batch):
        ids, masks, value, wdl = batch
        ids = [t.to(self.device) for t in ids]
        masks = [t.to(self.device) for t in masks]
        value = value.to(self.device)
        wdl = wdl.to(self.device)
        self.model.train()
        v_pred, w_pred = self.model(ids, masks)
        rank = None
        if self.cfg.get("lambda_rank", 0.0) > 0 and len(value) >= 4:
            idx = count_ranking_pairs(value.tolist(), self.cfg.get("rank_group", 4))
            if idx is not None:
                a, b, s = idx
                rank = (v_pred[a], v_pred[b], s.to(self.device))
        losses = self.loss_fn(v_pred, w_pred, value, wdl,
                              *(rank if rank is not None else (None, None, None)))
        self.opt.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.positions_seen += len(value)
        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def evaluate(self, records, batch_size=512):
        loader, _ = make_loader(records, batch_size=batch_size, shuffle=False)
        self.model.eval()
        tot = {"total": 0.0, "value": 0.0, "wdl": 0.0, "rank": 0.0}
        racc, wacc, n = 0.0, 0.0, 0
        nb = 0
        for batch in loader:
            ids, masks, value, wdl = batch
            v_pred, w_pred = self.model(ids, masks)
            losses = self.loss_fn(v_pred, w_pred, value, wdl)
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

    def fit(self, train_records, val_records, max_positions, batch_size=256, log_every=50,
            ckpt_dir=None, seed=0):
        loader, _ = make_loader(train_records, batch_size=batch_size, shuffle=True, seed=seed)
        history = []
        it = iter(loader)
        step = 0
        while self.positions_seen < max_positions:
            try:
                batch = next(it)
            except StopIteration:
                loader, _ = make_loader(train_records, batch_size=batch_size, shuffle=True,
                                        seed=seed + step)
                it = iter(loader)
                batch = next(it)
            losses = self.train_step(batch)
            step += 1
            if step % log_every == 0:
                val = self.evaluate(val_records)
                row = {"step": step, **losses, **{f"val_{k}": v for k, v in val.items()}}
                history.append(row)
        if ckpt_dir:
            self.save_checkpoint(ckpt_dir)
        return history

    def save_checkpoint(self, ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        meta = {
            "arch": self.cfg["arch"],
            "seed": self.cfg.get("seed", 0),
            "positions_seen": self.positions_seen,
            "params": self.model.parameter_count(),
            "config": self.cfg,
            "git_commit": os.environ.get("RUNE_GIT_COMMIT", "unknown"),
            "timestamp": time.time(),
        }
        with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
