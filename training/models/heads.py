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

from training.features.python_features import TOKEN_DIM, TOKENS, VOCAB_SIZES
from training.models.activations import clip01


class GroupedEmbedder(nn.Module):
    def __init__(self, vocab_sizes=None, token_dim=None):
        super().__init__()
        vocabs = list(vocab_sizes) if vocab_sizes is not None else list(VOCAB_SIZES)
        dim = token_dim if token_dim is not None else TOKEN_DIM
        self.vocabs = vocabs
        self.token_dim = dim
        self.num_groups = len(vocabs)
        self.tables = nn.ModuleList([nn.Embedding(v, dim) for v in vocabs])
        for emb in self.tables:
            nn.init.uniform_(emb.weight, -0.01, 0.01)

    def forward(self, group_ids, group_mask):
        last = self.num_groups - 1
        toks = []
        for g in range(last):
            e = self.tables[g](group_ids[g])
            e = (e * group_mask[g].unsqueeze(-1)).sum(dim=1)
            toks.append(e)
        e_last = self.tables[last](group_ids[last])
        e_last = (e_last * group_mask[last].unsqueeze(-1)).sum(dim=1)
        toks[0] = toks[0] + e_last
        x = torch.stack(toks, dim=1)
        return clip01(x)


class ValueWdlHead(nn.Module):
    def __init__(self, hidden1=128, hidden2=32, pair=False, input_dim=None):
        super().__init__()
        self.hidden1 = hidden1
        self.pair = pair
        in_dim = input_dim if input_dim is not None else TOKENS * TOKEN_DIM
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.fc2 = nn.Linear(hidden1 * 2 if pair else hidden1, hidden2)
        self.fcv = nn.Linear(hidden2, 1)
        self.fcwdl = nn.Linear(hidden2, 3)

    def forward(self, flat):
        pre = self.fc1(flat)
        c = clip01(pre)
        h1 = torch.cat([c, c * c], dim=-1) if self.pair else c
        h2 = clip01(self.fc2(h1))
        value = torch.tanh(self.fcv(h2)).squeeze(-1)
        wdl = self.fcwdl(h2)
        return value, wdl


class SfnnHead(nn.Module):
    def __init__(self, pair=False, input_dim=None):
        super().__init__()
        self.pair = pair
        in_dim = input_dim if input_dim is not None else TOKENS * TOKEN_DIM
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(512 if pair else 256, 32)
        self.fcv = nn.Linear(32, 1)
        self.fcwdl = nn.Linear(32, 3)

    def forward(self, flat):
        pre = self.fc1(flat)
        c = clip01(pre)
        h1 = torch.cat([c, c * c], dim=-1) if self.pair else c
        h2 = clip01(self.fc2(h1))
        value = torch.tanh(self.fcv(h2)).squeeze(-1)
        wdl = self.fcwdl(h2)
        return value, wdl


class BucketedHead(nn.Module):
    def __init__(self, head_fn=ValueWdlHead, pair=False, input_dim=None, phase_fn=None):
        super().__init__()
        self.pair = pair
        try:
            self.heads = nn.ModuleList([head_fn(pair=pair, input_dim=input_dim),
                                        head_fn(pair=pair, input_dim=input_dim),
                                        head_fn(pair=pair, input_dim=input_dim)])
        except TypeError:
            try:
                self.heads = nn.ModuleList([head_fn(pair=pair), head_fn(pair=pair), head_fn(pair=pair)])
            except TypeError:
                self.heads = nn.ModuleList([head_fn(), head_fn(), head_fn()])
        self.phase_fn = phase_fn

    def phases_from_ids(self, group_ids, group_mask):
        if self.phase_fn is not None:
            return self.phase_fn(group_ids, group_mask)
        g7 = group_ids[7].clamp(0, 63)
        valid = (group_mask[7] > 0.5) & (g7 >= 34) & (g7 <= 36)
        cand = torch.where(valid, g7, torch.full_like(g7, 34))
        return (cand.amax(dim=1) - 34).clamp(0, 2).long()

    def forward(self, flat, phase):
        value = torch.empty(flat.size(0), dtype=flat.dtype, device=flat.device)
        wdl = torch.empty(flat.size(0), 3, dtype=flat.dtype, device=flat.device)
        for b in range(3):
            rows = (phase == b).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            v, w = self.heads[b](flat[rows])
            value[rows] = v
            wdl[rows] = w
        return value, wdl