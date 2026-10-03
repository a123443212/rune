import torch
import torch.nn as nn

from training.losses.adaptive import AdaptiveLoss


class SearchLoss(nn.Module):
    def __init__(self, value=True, wdl=True, ranking=False, lambda_wdl=0.5,
                 lambda_rank=0.1, rank_margin=0.05, lambda_diff=0.1,
                 diff_margin=0.1, uncertainty=False, lambda_unc=0.2,
                 stability=False, lambda_stab=0.1):
        super().__init__()
        self.base = AdaptiveLoss(value, wdl, ranking, lambda_wdl,
                                 lambda_rank, rank_margin, lambda_diff,
                                 diff_margin)
        self.use_unc = uncertainty
        self.use_stab = stability
        self.lambda_unc = lambda_unc
        self.lambda_stab = lambda_stab
        self.mse = nn.MSELoss()

    def forward(self, cheap_v, cheap_w, ref_v, ref_w, diff, u, s_pred,
                value, wdl, rank=None, oracle=None, s_target=None,
                s_mask=None):
        parts = self.base(cheap_v, cheap_w, ref_v, ref_w, diff, value,
                          wdl, rank, oracle)
        total = parts["total"]
        if self.use_unc:
            with torch.no_grad():
                u_tgt = (value.detach() - ref_v.detach()).abs() / 2.0
                u_tgt = u_tgt.clamp(0.0, 1.0)
            parts["unc"] = self.mse(u, u_tgt)
            total = total + self.lambda_unc * parts["unc"]
        else:
            parts["unc"] = torch.zeros((), device=cheap_v.device)
        if self.use_stab and s_target is not None and s_mask is not None:
            m = s_mask.to(s_pred.device).float()
            if m.sum() > 0:
                parts["stab"] = ((s_pred - s_target.to(s_pred.device)) ** 2 * m).sum() / m.sum()
            else:
                parts["stab"] = torch.zeros((), device=cheap_v.device)
            total = total + self.lambda_stab * parts["stab"]
        else:
            parts["stab"] = torch.zeros((), device=cheap_v.device)
            parts["stab_skipped"] = torch.tensor(
                float(s_target is None), device=cheap_v.device)
        parts["total"] = total
        return parts
