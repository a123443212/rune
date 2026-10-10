# RUNE — Relational Unified Neural Evaluator
# Copyright (C) 2026 a123443212
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# This project is dual-licensed under the MIT License and the
# Apache License, Version 2.0. You may choose either license
# when using, copying, modifying, or distributing this software.
#
# MIT License: https://opensource.org/license/mit
# Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, this
# software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied.

import torch
import torch.nn as nn

from training.losses.components import RankingLoss, ValueLoss, WDLLoss


class AdaptiveLoss(nn.Module):
    def __init__(self, value=True, wdl=True, ranking=False, lambda_wdl=0.5,
                 lambda_rank=0.1, rank_margin=0.05, lambda_diff=0.1,
                 diff_margin=0.1):
        super().__init__()
        self.use_value = value
        self.use_wdl = wdl
        self.use_ranking = ranking
        self.lambda_wdl = lambda_wdl
        self.lambda_rank = lambda_rank
        self.lambda_diff = lambda_diff
        self.diff_margin = diff_margin
        self.value_loss = ValueLoss()
        self.wdl_loss = WDLLoss()
        self.rank_loss = RankingLoss(rank_margin)
        self.bce = nn.BCEWithLogitsLoss()

    def head_losses(self, v_pred, w_pred, value, wdl, prefix, parts, total):
        if self.use_value:
            parts[prefix + "value"] = self.value_loss(v_pred, value)
            total = total + parts[prefix + "value"]
        if self.use_wdl:
            parts[prefix + "wdl"] = self.wdl_loss(w_pred, wdl)
            total = total + self.lambda_wdl * parts[prefix + "wdl"]
        return total

    def forward(self, cheap_v, cheap_w, ref_v, ref_w, diff, value, wdl,
                rank=None, oracle=None):
        total = torch.zeros((), device=cheap_v.device)
        parts = {}
        total = self.head_losses(cheap_v, cheap_w, value, wdl, "cheap_", parts, total)
        total = self.head_losses(ref_v, ref_w, value, wdl, "ref_", parts, total)
        if self.use_ranking and rank is not None:
            parts["rank"] = self.rank_loss(*rank)
            total = total + self.lambda_rank * parts["rank"]
        else:
            parts["rank"] = torch.zeros((), device=cheap_v.device)
        if oracle is not None:
            target = oracle.to(diff.device).float()
            proxy = False
        else:
            with torch.no_grad():
                target = (ref_v.detach() - cheap_v.detach()).abs() > self.diff_margin
                target = target.float()
            proxy = True
        parts["diff"] = self.bce(diff, target)
        parts["diff_proxy"] = torch.tensor(float(proxy), device=cheap_v.device)
        total = total + self.lambda_diff * parts["diff"]
        parts["total"] = total
        return parts
