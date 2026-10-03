import torch
import torch.nn as nn

from training.models.adaptive import AdaptiveModel, route_mask

ARCH_ID = "RUNE-05"
ARCH_VERSION = "0.5.0"

ROUTING_MODES = ("difficulty", "uncertainty", "both", "full")


def route_multi(signal, mode, thresholds, prev=None):
    d = signal.get("difficulty", None)
    u = signal.get("uncertainty", None)
    s = signal.get("stability", None)
    t = thresholds
    if mode == "difficulty":
        return route_mask(d, t["difficulty"], t.get("t_low"), prev)
    if mode == "uncertainty":
        return route_mask(u, t["uncertainty"], None, None)
    need = route_mask(d, t["difficulty"], t.get("t_low"), prev)
    need_u = route_mask(u, t["uncertainty"], None, None)
    if mode == "both":
        return need | need_u
    need_s = route_mask(s, t["stability"], None, None)
    return need | need_u | need_s


class UncertaintyHead(nn.Module):
    def __init__(self, total_dim):
        super().__init__()
        self.fc = nn.Linear(total_dim, 1)

    def forward(self, flat):
        return torch.sigmoid(self.fc(flat)).squeeze(-1)

    def logit(self, flat):
        return self.fc(flat).squeeze(-1)


class StabilityHead(nn.Module):
    def __init__(self, total_dim):
        super().__init__()
        self.fc = nn.Linear(total_dim, 1)

    def forward(self, flat):
        return torch.clamp(self.fc(flat).squeeze(-1), min=0.0)


class SearchAwareModel(AdaptiveModel):
    def __init__(self, dim=32, cheap_pooling="none", alpha=1.0,
                 threshold=0.5, t_high=None, t_low=None, pruned_pairs=(),
                 refine_precision="fp32", uncertainty_on=True, stability_on=False):
        super().__init__(dim, cheap_pooling, alpha, threshold, t_high, t_low,
                         tuple(pruned_pairs), refine_precision)
        self.uncertainty_on = uncertainty_on
        self.stability_on = stability_on
        total = 8 * dim
        self.unc_head = UncertaintyHead(total)
        self.stab_head = StabilityHead(total)

    def search_forward(self, group_ids, group_mask):
        flat, cv, cw, diff = self.cheap_forward(group_ids, group_mask)
        rv, rw = self.refine_forward(flat)
        u = self.unc_head(flat.detach())
        s = self.stab_head(flat.detach())
        return cv, cw, rv, rw, diff, u, s

    def forward(self, group_ids, group_mask):
        return self.search_forward(group_ids, group_mask)

    def infer_search(self, group_ids, group_mask, routing="difficulty",
                     thresholds=None, prev=None):
        flat, cv, cw, diff = self.cheap_forward(group_ids, group_mask)
        u = self.unc_head(flat.detach())
        s = self.stab_head(flat.detach())
        t = {"difficulty": self.threshold, "uncertainty": 0.5,
             "stability": 0.5}
        if thresholds:
            t.update(thresholds)
        if routing not in ROUTING_MODES:
            raise ValueError(f"unknown routing {routing}")
        mask = route_multi({"difficulty": diff, "uncertainty": u, "stability": s},
                           routing, t, prev)
        rv, rw = self.refine_forward(flat)
        out_v = torch.where(mask, rv, cv)
        out_w = torch.where(mask.unsqueeze(-1).expand_as(rw), rw, cw)
        return out_v, out_w, mask, u, s

    def arch_tensors(self):
        d = super().arch_tensors()
        d["uw"] = self.unc_head.fc.weight.detach()
        d["ub"] = self.unc_head.fc.bias.detach()
        d["sw"] = self.stab_head.fc.weight.detach()
        d["sb"] = self.stab_head.fc.bias.detach()
        return d

    def export_order(self):
        return super().export_order() + ["uw", "ub", "sw", "sb"]

    def model_spec(self, quantization="fp32"):
        s = super().model_spec(quantization)
        s["arch"] = ARCH_ID
        s["arch_version"] = ARCH_VERSION
        s["uncertainty"] = self.uncertainty_on
        s["stability_head"] = self.stability_on
        return s


def build_search_model(dim=32, cheap_pooling="none", alpha=1.0, threshold=0.5,
                       t_high=None, t_low=None, pruned_pairs=(),
                       refine_precision="fp32", uncertainty_on=True,
                       stability_on=False):
    return SearchAwareModel(dim, cheap_pooling, alpha, threshold, t_high, t_low,
                            tuple(pruned_pairs), refine_precision, uncertainty_on,
                            stability_on)
