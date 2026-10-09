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


def count_ranking_pairs(values, group_size=4, tie_margin=0.0):
    n = (len(values) // group_size) * group_size
    if n < 2:
        return None
    a_idx, b_idx, signs = [], [], []
    for start in range(0, n, group_size):
        for j in range(group_size):
            for k in range(j + 1, group_size):
                i1, i2 = start + j, start + k
                if abs(values[i1] - values[i2]) < tie_margin:
                    continue
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
                                         dynamic_bias=p.get("dynamic_bias", False),
                                         pair=p.get("pair", config.get("pair", False)),
                                         game=config.get("game", "chess"))
            self.needs_context = True
            self.is_adaptive = False
            self.is_search = False
        elif config["arch"].startswith("RUNE-03-"):
            from training.models.dense import build_dense_model

            p = config.get("dense_params", {})
            self.model = build_dense_model(variant=config["arch"].split("-")[-1],
                                           token_dims=p.get("token_dims", [32] * 8),
                                           pooling=p.get("pooling", "none"),
                                           pool_clip=p.get("pool_clip", True),
                                           gate_on=p.get("gate_on", False),
                                           shared_width=p.get("shared_width", 32),
                                           head_h1=p.get("head_h1", 128),
                                           head_h2=p.get("head_h2", 32))
            self.needs_context = False
            self.is_adaptive = False
            self.is_search = False
        elif config["arch"].startswith("RUNE-04"):
            from training.models.adaptive import build_adaptive_model

            p = config.get("adaptive_params", {})
            self.model = build_adaptive_model(dim=p.get("dim", 32),
                                              cheap_pooling=p.get("cheap_pooling", "none"),
                                              alpha=p.get("alpha", 1.0),
                                              threshold=p.get("threshold", 0.5),
                                              t_high=p.get("t_high", None),
                                              t_low=p.get("t_low", None),
                                              pruned_pairs=p.get("pruned_pairs", ()),
                                              refine_precision=p.get("refine_precision", "fp32"),
                                              cheap_hidden=p.get("cheap_hidden", 32),
                                              ref_h1=p.get("ref_h1", 128),
                                              ref_h2=p.get("ref_h2", 32))
            self.needs_context = False
            self.is_adaptive = True
            self.is_search = False
        elif config["arch"].startswith("RUNE-05"):
            from training.models.uncertainty import build_search_model

            p = config.get("adaptive_params", {})
            self.model = build_search_model(dim=p.get("dim", 32),
                                            cheap_pooling=p.get("cheap_pooling", "none"),
                                            alpha=p.get("alpha", 1.0),
                                            threshold=p.get("threshold", 0.5),
                                            t_high=p.get("t_high", None),
                                            t_low=p.get("t_low", None),
                                            pruned_pairs=p.get("pruned_pairs", ()),
                                            refine_precision=p.get("refine_precision", "fp32"),
                                            uncertainty_on=True,
                                            stability_on=config.get("lambda_stab", 0.0) > 0,
                                            cheap_hidden=p.get("cheap_hidden", 32),
                                            ref_h1=p.get("ref_h1", 128),
                                            ref_h2=p.get("ref_h2", 32))
            self.needs_context = False
            self.is_adaptive = True
            self.is_search = True
        else:
            p0 = config.get("rel_params", {})
            self.model = build_model(config["arch"], gate=p0.get("gate", "clip"),
                                     pair=p0.get("pair", config.get("pair", False)),
                                     game=config.get("game", "chess"),
                                     buckets=config.get("head_buckets", 1))
            self.needs_context = False
            self.is_adaptive = False
            self.is_search = False
        if config["arch"].startswith("RUNE-05"):
            from training.losses.uncertainty import SearchLoss

            self.loss_fn = SearchLoss(
                value=True,
                wdl=config.get("lambda_wdl", 0.5) > 0,
                ranking=config.get("lambda_rank", 0.0) > 0,
                lambda_wdl=config.get("lambda_wdl", 0.5),
                lambda_rank=config.get("lambda_rank", 0.1),
                rank_margin=config.get("rank_margin", 0.05),
                lambda_diff=config.get("lambda_diff", 0.1),
                diff_margin=config.get("diff_margin", 0.1),
                uncertainty=config.get("lambda_unc", 0.0) > 0,
                lambda_unc=config.get("lambda_unc", 0.2),
                stability=config.get("lambda_stab", 0.0) > 0,
                lambda_stab=config.get("lambda_stab", 0.1),
            )
        elif config["arch"].startswith("RUNE-04"):
            from training.losses.adaptive import AdaptiveLoss

            self.loss_fn = AdaptiveLoss(
                value=True,
                wdl=config.get("lambda_wdl", 0.5) > 0,
                ranking=config.get("lambda_rank", 0.0) > 0,
                lambda_wdl=config.get("lambda_wdl", 0.5),
                lambda_rank=config.get("lambda_rank", 0.1),
                rank_margin=config.get("rank_margin", 0.05),
                lambda_diff=config.get("lambda_diff", 0.1),
                diff_margin=config.get("diff_margin", 0.1),
            )
        else:
            self.loss_fn = CompositeLoss(
                value=True,
                wdl=config.get("lambda_wdl", 0.5) > 0,
                ranking=config.get("lambda_rank", 0.0) > 0,
                lambda_wdl=config.get("lambda_wdl", 0.5),
                lambda_rank=config.get("lambda_rank", 0.1),
                rank_margin=config.get("rank_margin", 0.05),
            )
        dist = config.get("distill", {})
        self.is_distill = bool(dist.get("enabled", False))
        if self.is_distill:
            from training.losses.distillation import DistillCompositeLoss

            self.distill_fn = DistillCompositeLoss(
                task=dist.get("task", "value_wdl"),
                alpha=dist.get("alpha", 0.5),
                rank_margin=config.get("rank_margin", 0.05),
                lambda_wdl=config.get("lambda_wdl", 0.5),
                lambda_rank=config.get("lambda_rank", 0.1),
                weight_mode=dist.get("weight_mode", "uniform"),
                weight_lo=dist.get("weight_lo", 0.25),
                weight_hi=dist.get("weight_hi", 1.0),
                lambda_unc_distill=dist.get("lambda_unc_distill", 0.0),
                soft_wdl=dist.get("soft_wdl", False),
                quality_weighted=dist.get("quality_weighted", False),
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
        if len(batch) == 6:
            ids, masks, ctx, value, wdl, teach = batch
            return ids, masks, ctx, value, wdl, teach
        if len(batch) == 5:
            ids, masks, ctx, value, wdl = batch
            return ids, masks, ctx, value, wdl, None
        ids, masks, value, wdl = batch[:4]
        return ids, masks, None, value, wdl, None

    def student_outputs(self, out):
        if len(out) == 7:
            return out[2], out[3], out[5]
        if len(out) == 5:
            return out[2], out[3], None
        return out[0], out[1], None

    def forward_model(self, ids, masks, ctx):
        ids = [t.to(self.device) for t in ids]
        masks = [t.to(self.device) for t in masks]
        if self.needs_context:
            return self.model(ids, masks, ctx.to(self.device))
        return self.model(ids, masks)

    def train_step(self, batch, rank_batch=None):
        t0 = time.time()
        ids, masks, ctx, value, wdl, teach = self.unpack(batch)
        value = value.to(self.device)
        wdl = wdl.to(self.device)
        self.model.train()
        out = self.forward_model(ids, masks, ctx)
        rank = None
        if rank_batch is not None:
            rids, rmasks, rctx, a_idx, b_idx, signs, weights = rank_batch
            r_out = self.forward_model(rids, rmasks, rctx)
            if self.is_adaptive:
                rv_pred = r_out[2]
            else:
                rv_pred, _ = r_out
            a = torch.tensor(a_idx, device=self.device)
            b = torch.tensor(b_idx, device=self.device)
            rank = (rv_pred[a], rv_pred[b], torch.tensor(signs, device=self.device),
                    torch.tensor(weights, device=self.device))
        elif self.cfg.get("lambda_rank", 0.0) > 0 and len(value) >= 4:
            idx = count_ranking_pairs(value.tolist(), self.cfg.get("rank_group", 4),
                                  self.cfg.get("rank_tie_margin", 0.0))
            if idx is not None:
                a, b, s = idx
                base = out[2] if self.is_adaptive else out[0]
                rank = (base[a], base[b], s.to(self.device))
        if self.is_search:
            cheap_v, cheap_w, ref_v, ref_w, diff, u, s_pred = out
            losses = self.loss_fn(cheap_v, cheap_w, ref_v, ref_w, diff, u, s_pred,
                                  value, wdl, rank)
        elif self.is_adaptive:
            cheap_v, cheap_w, ref_v, ref_w, diff = out
            losses = self.loss_fn(cheap_v, cheap_w, ref_v, ref_w, diff,
                                  value, wdl, rank)
        else:
            v_pred, w_pred = out
            losses = self.loss_fn(v_pred, w_pred, value, wdl, rank)
        if self.is_distill and teach is not None:
            import torch.nn.functional as F

            sv, sw, su = self.student_outputs(out)
            tv = teach["v"].to(self.device)
            tw = teach["w"].to(self.device)
            tu = teach["u"].to(self.device)
            dlosses = self.distill_fn(sv, sw, tv, F.one_hot(tw, 3).float(), value,
                                      wdl, teacher_u=tu, student_u=su, rank=rank,
                                      teacher_probs=teach.get("wp", None),
                                      quality=teach.get("qw", None))
            losses = dict(losses)
            for k, v in dlosses.items():
                losses[k if k != "total" else "dist_total"] = v
            losses["total"] = losses["total"] + dlosses["total"]
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
        tot = {"total": 0.0, "value": 0.0, "wdl": 0.0, "rank": 0.0,
               "cheap_value": 0.0, "cheap_wdl": 0.0, "ref_value": 0.0,
               "ref_wdl": 0.0, "diff": 0.0, "unc": 0.0, "stab": 0.0,
               "task_value": 0.0, "task_wdl": 0.0, "distill": 0.0,
               "distill_wdl": 0.0, "distill_wmean": 0.0, "unc_distill": 0.0,
               "dist_total": 0.0}
        racc, wacc, n = 0.0, 0.0, 0
        nb = 0
        cwacc, cn, dsum = 0.0, 0, 0.0
        usum, esum, uusum, eesum, uesum, qn = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        tsum, tsqn, ttn = 0.0, 0, 0
        for batch in loader:
            ids, masks, ctx, value, wdl, teach = self.unpack(batch)
            out = self.forward_model(ids, masks, ctx)
            if self.is_search:
                cheap_v, cheap_w, v_pred, w_pred, diff, u, _ = out
                losses = self.loss_fn(cheap_v, cheap_w, v_pred, w_pred, diff, u, None,
                                      value.to(self.device), wdl.to(self.device))
                cwacc += wdl_accuracy(cheap_w, wdl).item() * len(value)
                cn += len(value)
                dsum += diff.float().mean().item() * len(value)
                with torch.no_grad():
                    ee = (value.to(self.device) - v_pred).abs() / 2.0
                    uu = u.detach().float()
                    usum += uu.sum().item()
                    esum += ee.sum().item()
                    uusum += (uu * uu).sum().item()
                    eesum += (ee * ee).sum().item()
                    uesum += (uu * ee).sum().item()
                    qn += len(value)
            elif self.is_adaptive:
                cheap_v, cheap_w, v_pred, w_pred, diff = out
                losses = self.loss_fn(cheap_v, cheap_w, v_pred, w_pred, diff,
                                      value.to(self.device), wdl.to(self.device))
                cwacc += wdl_accuracy(cheap_w, wdl).item() * len(value)
                cn += len(value)
                dsum += diff.float().mean().item() * len(value)
            else:
                v_pred, w_pred = out
                losses = self.loss_fn(v_pred, w_pred, value.to(self.device), wdl.to(self.device))
            if self.is_distill and teach is not None:
                import torch.nn.functional as _F

                sv, sw, su = self.student_outputs(out)
                tv = teach["v"].to(self.device)
                tw = teach["w"].to(self.device)
                tu = teach["u"].to(self.device)
                dlosses = self.distill_fn(sv, sw, tv, _F.one_hot(tw, 3).float(),
                                          value.to(self.device), wdl.to(self.device),
                                          teacher_u=tu, student_u=su,
                                          teacher_probs=teach.get("wp", None),
                                          quality=teach.get("qw", None))
                for k, v in dlosses.items():
                    losses[k if k != "total" else "dist_total"] = v
            if teach is not None:
                with torch.no_grad():
                    tv = teach["v"].to(self.device)
                    sv, _, _ = self.student_outputs(out)
                    tsum += (sv.detach() - tv).abs().sum().item()
                    tsqn += ((tv - value.to(self.device)).abs()).sum().item()
                    ttn += len(value)
            for k in tot:
                if k in losses:
                    tot[k] += losses[k].item()
            wacc += wdl_accuracy(w_pred, wdl).item() * len(value)
            if len(value) >= 4:
                idx = count_ranking_pairs(value.tolist(), 4,
                                          self.cfg.get("rank_tie_margin", 0.0))
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
        if self.is_adaptive:
            out["cheap_wdl_acc"] = cwacc / max(1, cn)
            out["difficulty_mean"] = dsum / max(1, cn)
            out["cheap_params"] = self.model.cheap_parameter_count()
        if self.is_search:
            denom = (qn * uusum - usum * usum) * (qn * eesum - esum * esum)
            out["unc_err_corr"] = (qn * uesum - usum * esum) / max(1e-12, denom ** 0.5) \
                if denom > 0 else 0.0
            out["unc_mean"] = usum / max(1, qn)
        if ttn > 0:
            out["student_vs_teacher_mae"] = tsum / ttn
            out["teacher_vs_target_mae"] = tsqn / ttn
        return out

    def make_train_loader(self, records, batch_size, shuffle, seed):
        game = self.cfg.get("game", "chess")
        if self.needs_context:
            from training.datasets.flex_dataset import make_flex_loader

            return make_flex_loader(records, batch_size=batch_size, shuffle=shuffle, seed=seed, game=game)
        return make_loader(records, batch_size=batch_size, shuffle=shuffle, seed=seed, game=game)

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
