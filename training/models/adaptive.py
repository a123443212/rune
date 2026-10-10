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

from training.models.dense import ChannelGate, TokenPool, VarEmbedder

TOKENS = 8
ARCH_ID = "RUNE-04"
ARCH_VERSION = "0.4.0"
FEATURE_SET = "grouped_hkav2_fullthreats_v02"


def wdl_entropy(wdl_logits):
    p = F.softmax(wdl_logits, dim=-1).clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1)


def route_mask(difficulty, threshold, t_low=None, prev=None):
    d = difficulty.detach() if isinstance(difficulty, torch.Tensor) else difficulty
    if t_low is not None and prev is not None:
        refine = torch.where(prev, d >= t_low, d >= threshold)
        return refine
    return d >= threshold


class CheapHead(nn.Module):
    def __init__(self, total_dim, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(total_dim, hidden)
        self.fcv = nn.Linear(hidden, 1)
        self.fcwdl = nn.Linear(hidden, 3)

    def forward(self, flat):
        h = torch.clamp(self.fc1(flat), 0.0, 1.0)
        value = torch.tanh(self.fcv(h)).squeeze(-1)
        return value, self.fcwdl(h)


class DifficultyHead(nn.Module):
    def __init__(self, total_dim):
        super().__init__()
        self.fc = nn.Linear(total_dim, 1)

    def forward(self, flat):
        return self.fc(flat).squeeze(-1)


class RefinementBlock(nn.Module):
    def __init__(self, tokens=TOKENS, dim=32, alpha=1.0, pruned_pairs=()):
        super().__init__()
        self.tokens = tokens
        self.dim = dim
        self.alpha = alpha
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.gab = nn.Parameter(torch.zeros(tokens, tokens))
        mask = torch.ones(tokens, tokens)
        for i, j in pruned_pairs:
            mask[i, j] = 0.0
        self.register_buffer("prune_mask", mask)

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        s = q @ k.transpose(-1, -2) + self.gab
        g = torch.clamp(s, 0.0, 1.0) * self.prune_mask
        return x + self.alpha * (g @ v)


class RefinedHead(nn.Module):
    def __init__(self, total_dim, h1=128, h2=32):
        super().__init__()
        self.h1 = h1
        self.h2 = h2
        self.fc1 = nn.Linear(total_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fcv = nn.Linear(h2, 1)
        self.fcwdl = nn.Linear(h2, 3)

    def forward(self, flat):
        h1 = torch.clamp(self.fc1(flat), 0.0, 1.0)
        h2 = torch.clamp(self.fc2(h1), 0.0, 1.0)
        value = torch.tanh(self.fcv(h2)).squeeze(-1)
        return value, self.fcwdl(h2)


class AdaptiveModel(nn.Module):
    def __init__(self, dim=32, cheap_pooling="none", alpha=1.0,
                 threshold=0.5, t_high=None, t_low=None, pruned_pairs=(),
                 refine_precision="fp32", cheap_hidden=32, ref_h1=128,
                 ref_h2=32):
        super().__init__()
        if cheap_pooling not in ("none", "shared"):
            raise ValueError(f"unknown cheap pooling {cheap_pooling}")
        if cheap_pooling == "shared" and dim != 32:
            raise ValueError("shared cheap pooling requires dim 32")
        self.dim = dim
        self.cheap_pooling = cheap_pooling
        self.threshold = threshold
        self.t_high = threshold if t_high is None else t_high
        self.t_low = t_low
        if self.t_low is not None and not (self.t_low <= self.threshold <= self.t_high):
            raise ValueError(f"need t_low <= threshold <= t_high, got {self.t_low}, {self.threshold}, {self.t_high}")
        self.refine_precision = refine_precision
        self.cheap_hidden = cheap_hidden
        self.ref_h1 = ref_h1
        self.ref_h2 = ref_h2
        widths = [dim] * TOKENS
        self.embedder = VarEmbedder(widths)
        self.pool = TokenPool(widths, cheap_pooling, True, 32)
        self.gate = ChannelGate(widths, False)
        total = TOKENS * dim
        self.cheap_head = CheapHead(total, cheap_hidden)
        self.difficulty = DifficultyHead(total)
        self.refine = RefinementBlock(TOKENS, dim, alpha, tuple(pruned_pairs))
        self.refined_head = RefinedHead(total, ref_h1, ref_h2)

    def cheap_forward(self, group_ids, group_mask):
        acc = self.embedder(group_ids, group_mask)
        segs = self.pool.split(acc, self.embedder.group_widths)
        flat = torch.cat(self.pool(segs), dim=1)
        flat = self.gate(flat)
        value, wdl = self.cheap_head(flat)
        diff = self.difficulty(flat.detach())
        return flat, value, wdl, diff

    def refine_forward(self, flat):
        b = flat.size(0)
        x = flat.reshape(b, TOKENS, self.dim)
        y = self.refine(x)
        return self.refined_head(y.reshape(b, -1))

    def forward(self, group_ids, group_mask):
        flat, cv, cw, diff = self.cheap_forward(group_ids, group_mask)
        rv, rw = self.refine_forward(flat)
        return cv, cw, rv, rw, diff

    def infer(self, group_ids, group_mask, mode="adaptive", threshold=None,
              t_low=None, prev=None):
        flat, cv, cw, diff = self.cheap_forward(group_ids, group_mask)
        t = self.threshold if threshold is None else threshold
        if mode == "cheap":
            return cv, cw, torch.zeros_like(diff, dtype=torch.bool)
        if mode == "always":
            rv, rw = self.refine_forward(flat)
            return rv, rw, torch.ones_like(diff, dtype=torch.bool)
        mask = route_mask(diff, t, t_low if t_low is not None else self.t_low, prev)
        rv, rw = self.refine_forward(flat)
        out_v = torch.where(mask, rv, cv)
        out_w = torch.where(mask.unsqueeze(-1).expand_as(rw), rw, cw)
        return out_v, out_w, mask

    def arch_tensors(self):
        d = {}
        if self.cheap_pooling == "shared":
            d["pool_S"] = self.pool.shared_s.detach()
            for t in range(TOKENS):
                d[f"pool_s{t}"] = self.pool.tok_s[t].detach()
                d[f"pool_b{t}"] = self.pool.tok_b[t].detach()
        d.update({
            "cw1": self.cheap_head.fc1.weight.detach(),
            "cb1": self.cheap_head.fc1.bias.detach(),
            "cwv": self.cheap_head.fcv.weight.detach(),
            "cbv": self.cheap_head.fcv.bias.detach(),
            "cww": self.cheap_head.fcwdl.weight.detach(),
            "cbw": self.cheap_head.fcwdl.bias.detach(),
            "dw": self.difficulty.fc.weight.detach(),
            "db": self.difficulty.fc.bias.detach(),
            "wq": self.refine.wq.weight.detach(),
            "bq": self.refine.wq.bias.detach(),
            "wk": self.refine.wk.weight.detach(),
            "bk": self.refine.wk.bias.detach(),
            "wvv": self.refine.wv.weight.detach(),
            "bvv": self.refine.wv.bias.detach(),
            "gabS": self.refine.gab.detach(),
            "w1": self.refined_head.fc1.weight.detach(),
            "b1": self.refined_head.fc1.bias.detach(),
            "w2": self.refined_head.fc2.weight.detach(),
            "b2": self.refined_head.fc2.bias.detach(),
            "wvo": self.refined_head.fcv.weight.detach(),
            "bvo": self.refined_head.fcv.bias.detach(),
            "wwdl": self.refined_head.fcwdl.weight.detach(),
            "bwdl": self.refined_head.fcwdl.bias.detach(),
        })
        return d

    def export_order(self):
        order = []
        if self.cheap_pooling == "shared":
            order += ["pool_S"]
            for t in range(TOKENS):
                order += [f"pool_s{t}", f"pool_b{t}"]
        return order + ["cw1", "cb1", "cwv", "cbv", "cww", "cbw", "dw", "db",
                        "wq", "bq", "wk", "bk", "wvv", "bvv", "gabS",
                        "w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"]

    def embedding_tensors(self):
        return {f"emb{g}": self.embedder.tables[g].weight.detach() for g in range(9)}

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def cheap_parameter_count(self):
        n = sum(p.numel() for p in self.embedder.parameters())
        n += self.pool.parameterCount() if hasattr(self.pool, "parameterCount") else sum(
            p.numel() for p in self.pool.parameters())
        n += sum(p.numel() for p in self.cheap_head.parameters())
        n += sum(p.numel() for p in self.difficulty.parameters())
        return n

    def model_size_bytes(self):
        return self.parameter_count() * 4

    def model_spec(self, quantization="fp32"):
        return {
            "arch": ARCH_ID,
            "arch_version": ARCH_VERSION,
            "feature_set": FEATURE_SET,
            "tokens": TOKENS,
            "token_dim": self.dim,
            "token_dims": [self.dim] * TOKENS,
            "attention": "static_refinement",
            "geometric_bias": "static",
            "head": "cheap_plus_refined",
            "quantization": quantization,
            "cheap_pooling": self.cheap_pooling,
            "threshold": self.threshold,
            "t_high": self.t_high,
            "t_low": self.t_low,
            "refine_precision": self.refine_precision,
            "pruned_pairs": [list(p) for p in self.refine.prune_mask.eq(0).nonzero().tolist()],
            "cheap_hidden": self.cheap_hidden,
            "ref_h1": self.ref_h1,
            "ref_h2": self.ref_h2,
        }


def build_adaptive_model(dim=32, cheap_pooling="none", alpha=1.0, threshold=0.5,
                         t_high=None, t_low=None, pruned_pairs=(),
                         refine_precision="fp32", cheap_hidden=32, ref_h1=128,
                         ref_h2=32):
    return AdaptiveModel(dim, cheap_pooling, alpha, threshold, t_high, t_low,
                         tuple(pruned_pairs), refine_precision, cheap_hidden,
                         ref_h1, ref_h2)
