import torch
import torch.nn as nn

from training.losses.components import RankingLoss, ValueLoss, WDLLoss


class CompositeLoss(nn.Module):
    def __init__(self, value=True, wdl=True, ranking=False, lambda_wdl=0.5,
                 lambda_rank=0.1, rank_margin=0.05):
        super().__init__()
        self.use_value = value
        self.use_wdl = wdl
        self.use_ranking = ranking
        self.lambda_wdl = lambda_wdl
        self.lambda_rank = lambda_rank
        self.value_loss = ValueLoss()
        self.wdl_loss = WDLLoss()
        self.rank_loss = RankingLoss(rank_margin)

    def forward(self, value_pred, wdl_pred, value_tgt, wdl_tgt, rank=None):
        total = torch.zeros((), device=value_pred.device)
        parts = {}
        if self.use_value:
            parts["value"] = self.value_loss(value_pred, value_tgt)
            total = total + parts["value"]
        if self.use_wdl:
            parts["wdl"] = self.wdl_loss(wdl_pred, wdl_tgt)
            total = total + self.lambda_wdl * parts["wdl"]
        if self.use_ranking and rank is not None:
            parts["rank"] = self.rank_loss(*rank)
            total = total + self.lambda_rank * parts["rank"]
        else:
            parts["rank"] = torch.zeros((), device=value_pred.device)
        parts["total"] = total
        return parts
