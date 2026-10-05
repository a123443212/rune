import torch
import torch.nn as nn
import torch.nn.functional as F


def confidence_weights(teacher_unc, mode="uniform", lo=0.25, hi=1.0):
    if mode == "uniform":
        return torch.ones_like(teacher_unc)
    u = teacher_unc.clamp(0.0, 1.0)
    if mode == "confidence":
        w = 1.0 - u
    elif mode == "difficulty":
        w = u
    else:
        raise ValueError(f"unknown weighting {mode}")
    return lo + (hi - lo) * w


class DistillationLoss(nn.Module):
    def forward(self, student_value, teacher_value):
        return F.mse_loss(student_value, teacher_value)


class WeightedDistillationLoss(nn.Module):
    def forward(self, student_value, teacher_value, weights):
        w = weights.clamp_min(0.0)
        return ((student_value - teacher_value) ** 2 * w).sum() / w.sum().clamp_min(1e-12)


class DistillCompositeLoss(nn.Module):
    def __init__(self, task="value_wdl", alpha=0.5, rank_margin=0.05,
                 lambda_wdl=0.5, lambda_rank=0.1, weight_mode="uniform",
                 weight_lo=0.25, weight_hi=1.0, lambda_unc_distill=0.0,
                 soft_wdl=False, quality_weighted=False):
        super().__init__()
        from training.losses.components import RankingLoss, ValueLoss, WDLLoss

        self.task = task
        self.alpha = alpha
        self.lambda_wdl = lambda_wdl
        self.lambda_rank = lambda_rank
        self.lambda_unc_distill = lambda_unc_distill
        self.weight_mode = weight_mode
        self.weight_lo = weight_lo
        self.weight_hi = weight_hi
        self.soft_wdl = soft_wdl
        self.quality_weighted = quality_weighted
        self.value_loss = ValueLoss()
        self.wdl_loss = WDLLoss()
        self.rank_loss = RankingLoss(rank_margin)
        self.distill_loss = DistillationLoss()
        self.wdistill_loss = WeightedDistillationLoss()
        self.mse = nn.MSELoss()

    def forward(self, student_v, student_w, teacher_v, teacher_w, value, wdl,
                teacher_u=None, student_u=None, rank=None, teacher_rank=None,
                teacher_probs=None, quality=None):
        parts = {}
        total = torch.zeros((), device=student_v.device)
        if self.task in ("value_wdl", "value"):
            parts["task_value"] = self.value_loss(student_v, value)
            total = total + parts["task_value"]
        else:
            parts["task_value"] = torch.zeros((), device=student_v.device)
        if self.task == "value_wdl":
            parts["task_wdl"] = self.wdl_loss(student_w, wdl)
            total = total + self.lambda_wdl * parts["task_wdl"]
        else:
            parts["task_wdl"] = torch.zeros((), device=student_v.device)
        if teacher_v is not None:
            use_quality = self.quality_weighted and quality is not None
            use_signal = self.weight_mode != "uniform" and teacher_u is not None
            if not use_quality and not use_signal:
                parts["distill"] = self.distill_loss(student_v, teacher_v)
                parts["distill_wmean"] = torch.ones((), device=student_v.device)
            else:
                w = confidence_weights(teacher_u if teacher_u is not None else torch.zeros_like(teacher_v),
                                       self.weight_mode, self.weight_lo, self.weight_hi)
                if use_quality:
                    w = w * quality.to(w.device).clamp(0.0, 1.0)
                parts["distill"] = self.wdistill_loss(student_v, teacher_v, w)
                parts["distill_wmean"] = w.mean()
            total = total + self.alpha * parts["distill"]
        else:
            parts["distill"] = torch.zeros((), device=student_v.device)
            parts["distill_wmean"] = torch.zeros((), device=student_v.device)
        if teacher_w is not None and self.task == "value_wdl":
            if self.soft_wdl and teacher_probs is not None:
                logp = F.log_softmax(student_w, dim=-1)
                parts["distill_wdl"] = -(teacher_probs.to(logp.device) * logp).sum(-1).mean()
            else:
                parts["distill_wdl"] = self.wdl_loss(student_w, teacher_w.argmax(dim=-1))
            total = total + self.alpha * parts["distill_wdl"]
        else:
            parts["distill_wdl"] = torch.zeros((), device=student_v.device)
        use_rank = rank if rank is not None else teacher_rank
        if use_rank is not None:
            parts["rank"] = self.rank_loss(*use_rank)
            total = total + self.lambda_rank * parts["rank"]
        else:
            parts["rank"] = torch.zeros((), device=student_v.device)
        if teacher_u is not None and student_u is not None and self.lambda_unc_distill > 0:
            parts["unc_distill"] = self.mse(student_u, teacher_u.detach())
            total = total + self.lambda_unc_distill * parts["unc_distill"]
        else:
            parts["unc_distill"] = torch.zeros((), device=student_v.device)
        parts["total"] = total
        return parts
