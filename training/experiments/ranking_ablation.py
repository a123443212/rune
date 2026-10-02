import json
import os

import torch

from training.datasets.flex_dataset import FlexDataset
from training.datasets.rune_dataset import RuneDataset
from training.datasets.siblings import pairs_to_batch
from training.losses.losses import ranking_accuracy
from training.trainer.trainer import Trainer


def chunked(items, n):
    return [items[i:i + n] for i in range(0, len(items), n)]


class RankingAblation:
    def __init__(self, config):
        self.cfg = config

    def base_trainer_cfg(self, use_rank):
        return {
            "arch": self.cfg["arch"],
            "seed": self.cfg.get("seed", 0),
            "lr": self.cfg.get("lr", 3e-4),
            "weight_decay": 0.01,
            "lambda_wdl": 0.5,
            "lambda_rank": self.cfg.get("lambda_rank", 0.1) if use_rank else 0.0,
            "rel_params": self.cfg.get("rel_params", {}),
        }

    @torch.no_grad()
    def sibling_accuracy(self, trainer, dataset, pairs):
        trainer.model.eval()
        correct, total = 0, 0
        for chunk in chunked(pairs, 32):
            ids, masks, a, b, signs, _ = pairs_to_batch(dataset, chunk)
            ctx = self._ctx_for(dataset, chunk)
            trainer.model.eval()
            if trainer.needs_context:
                v, _ = trainer.model(ids, masks, ctx)
            else:
                v, _ = trainer.model(ids, masks)
            a_t = torch.tensor(a)
            b_t = torch.tensor(b)
            s_t = torch.tensor(signs)
            correct += (torch.sign(v[a_t] - v[b_t]) == s_t).sum().item()
            total += len(signs)
        return correct / max(1, total)

    def _ctx_for(self, dataset, chunk):
        from training.features.context import context_vector

        fens = []
        for p in chunk:
            fens += [p["child_a"], p["child_b"]]
        return torch.tensor([context_vector(f) for f in fens], dtype=torch.float32)

    def run(self, train_records, val_records, pairs, out_dir, budget):
        os.makedirs(out_dir, exist_ok=True)
        flex = self.cfg["arch"] == "RUNE-REL-02"
        ds_cls = FlexDataset if flex else RuneDataset
        recs = [{"fen": p["child_a"], "value": 0.0, "wdl": 1} for p in pairs]
        recs += [{"fen": p["child_b"], "value": 0.0, "wdl": 1} for p in pairs]
        probe_ds = ds_cls(recs)
        results = {}
        for tag, use_rank in (("L0", False), ("L1", True)):
            trainer = Trainer(self.base_trainer_cfg(use_rank))
            main_ds = ds_cls(train_records)
            rank_batches = None
            if use_rank:
                rank_batches = []
                for chunk in chunked(pairs, 32):
                    ids, masks, a, b, signs, weights = pairs_to_batch(main_ds, chunk)
                    if flex:
                        ctx = self._ctx_for(main_ds, chunk)
                        rank_batches.append((ids, masks, ctx, a, b, signs, weights))
                    else:
                        rank_batches.append((ids, masks, None, a, b, signs, weights))
            ckpt = os.path.join(out_dir, tag)
            trainer.fit(train_records, val_records, budget,
                        batch_size=self.cfg.get("batch_size", 32), log_every=10**9,
                        ckpt_dir=ckpt, seed=self.cfg.get("seed", 0),
                        rank_batches=rank_batches)
            val = trainer.evaluate(val_records)
            val["sibling_acc"] = self.sibling_accuracy(trainer, probe_ds, pairs)
            results[tag] = val
        with open(os.path.join(out_dir, "ranking_ablation.json"), "w") as f:
            json.dump(results, f, indent=2)
        return results
