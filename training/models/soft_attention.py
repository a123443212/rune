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

import math

import torch
import torch.nn as nn

from training.features.python_features import TOKEN_DIM, TOKENS


class RuneSoftAttention(nn.Module):
    def __init__(self, use_gab=True, tokens=None, token_dim=None):
        super().__init__()
        t = tokens if tokens is not None else TOKENS
        d = token_dim if token_dim is not None else TOKEN_DIM
        self.use_gab = use_gab
        self.tokens_n = t
        self.token_dim_n = d
        self.scale = 1.0 / math.sqrt(d)
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.gab = nn.Parameter(torch.zeros(t, t))

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        scores = (q @ k.transpose(-1, -2)) * self.scale
        if self.use_gab:
            scores = scores + self.gab
        weights = torch.softmax(scores, dim=-1)
        return x + weights @ v
