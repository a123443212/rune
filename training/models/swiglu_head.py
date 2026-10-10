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

from training.features.python_features import TOKEN_DIM, TOKENS
from training.models.activations import clip01


class SwiGluWdlHead(nn.Module):
    def __init__(self, hidden=128, out_hidden=32, input_dim=None):
        super().__init__()
        in_dim = input_dim if input_dim is not None else TOKENS * TOKEN_DIM
        self.hidden = hidden
        self.out_hidden = out_hidden
        self.fc_gate = nn.Linear(in_dim, hidden)
        self.fc_up = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_hidden)
        self.fcv = nn.Linear(out_hidden, 1)
        self.fcwdl = nn.Linear(out_hidden, 3)

    def forward(self, flat):
        g = self.fc_gate(flat)
        h = (g * torch.sigmoid(g)) * self.fc_up(flat)
        h2 = clip01(self.fc2(h))
        value = torch.tanh(self.fcv(h2)).squeeze(-1)
        wdl = self.fcwdl(h2)
        return value, wdl
