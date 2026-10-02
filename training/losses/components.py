import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueLoss(nn.Module):
    def forward(self, pred, target):
        return F.mse_loss(pred, target)


class WDLLoss(nn.Module):
    def forward(self, logits, target):
        return F.cross_entropy(logits, target)


class RankingLoss(nn.Module):
    def __init__(self, margin=0.05):
        super().__init__()
        self.margin = margin

    def forward(self, pred_a, pred_b, sign, weight=None):
        diff = (pred_a - pred_b) * sign
        per = torch.clamp(self.margin - diff, min=0.0)
        if weight is not None:
            per = per * weight
        return per.mean()
