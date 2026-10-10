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
