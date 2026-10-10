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
