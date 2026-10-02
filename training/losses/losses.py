import torch
import torch.nn as nn
import torch.nn.functional as F


class RuneLoss(nn.Module):
    def __init__(self, lambda_wdl=0.5, lambda_rank=0.1, rank_margin=0.05):
        super().__init__()
        self.lambda_wdl = lambda_wdl
        self.lambda_rank = lambda_rank
        self.rank_margin = rank_margin

    def forward(self, value_pred, wdl_pred, value_tgt, wdl_tgt, rank_a=None, rank_b=None, rank_sign=None):
        l_value = F.mse_loss(value_pred, value_tgt)
        l_wdl = F.cross_entropy(wdl_pred, wdl_tgt)
        if rank_a is not None and rank_b is not None and rank_sign is not None:
            diff = (rank_a - rank_b) * rank_sign
            l_rank = torch.clamp(self.rank_margin - diff, min=0.0).mean()
        else:
            l_rank = torch.zeros((), device=value_pred.device)
        total = l_value + self.lambda_wdl * l_wdl + self.lambda_rank * l_rank
        return {
            "total": total,
            "value": l_value.detach(),
            "wdl": l_wdl.detach(),
            "rank": l_rank.detach(),
        }


def ranking_accuracy(value_pred_a, value_pred_b, sign):
    pred_sign = torch.sign(value_pred_a - value_pred_b)
    return ((pred_sign == sign.float()).float()).mean()


def wdl_accuracy(wdl_logits, wdl_tgt):
    return (wdl_logits.argmax(dim=-1) == wdl_tgt).float().mean()
