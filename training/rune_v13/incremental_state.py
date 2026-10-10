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

from training.rune_v13.interaction_graph import GRAPH_VERSION, INVALIDATION_VERSION
from training.rune_v13.relational_reference import DYN_CAP, split_qkv_score_gate_mix


def _gate_fn(name, s):
    if name == "hard_sigmoid":
        return torch.clamp(0.2 * s + 0.5, 0.0, 1.0)
    return torch.clamp(s, 0.0, 1.0)


class PathSelector:
    def __init__(self, threshold=2):
        self.threshold = int(threshold)

    def use_incremental(self, changed_tokens):
        return len(changed_tokens) <= self.threshold


class IncrementalRelationalState:
    def __init__(self, wq, bq, wk, bk, wv, bv, gab, gate="clip", alpha=1.0, threshold=2):
        self.wq = wq
        self.bq = bq
        self.wk = wk
        self.bk = bk
        self.wv = wv
        self.bv = bv
        self.gab = gab
        self.gate = gate
        self.alpha = float(alpha)
        self.selector = PathSelector(threshold)
        self.graph_version = GRAPH_VERSION
        self.invalidation_version = INVALIDATION_VERSION
        self.tokens = None
        self.cache = None
        self.ctx = None
        self.dyn_u = None
        self.dyn_w = None
        self.fallbacks = 0
        self.incremental_updates = 0
        self._stack = []

    def bind_dynamic(self, dyn_u, dyn_w):
        self.dyn_u = dyn_u
        self.dyn_w = dyn_w

    def rebuild(self, tokens, ctx=None):
        self.tokens = tokens.clone()
        self.ctx = None if ctx is None else ctx.clone()
        self.cache = split_qkv_score_gate_mix(
            tokens, self.wq, self.bq, self.wk, self.bk, self.wv, self.bv,
            self.gab, self.gate, self.alpha, self.ctx, self.dyn_u, self.dyn_w)
        self._stack = []
        return self.cache["out"]

    def push(self):
        assert self.cache is not None
        self._stack.append({
            "tokens": self.tokens.clone(),
            "q": self.cache["q"].clone(),
            "k": self.cache["k"].clone(),
            "v": self.cache["v"].clone(),
            "scores": self.cache["scores"].clone(),
            "gates": self.cache["gates"].clone(),
            "mixed": self.cache["mixed"].clone(),
            "out": self.cache["out"].clone(),
        })

    def pop(self):
        s = self._stack.pop()
        self.tokens = s["tokens"]
        for key in ("q", "k", "v", "scores", "gates", "mixed", "out"):
            self.cache[key] = s[key]

    def update(self, tokens_new, changed_tokens, ctx=None):
        assert self.cache is not None
        if not self.selector.use_incremental(changed_tokens):
            self.fallbacks += 1
            return self.rebuild(tokens_new, ctx)
        self.incremental_updates += 1
        changed = sorted(int(t) for t in changed_tokens)
        cset = set(changed)
        t = tokens_new.size(0)
        q = self.cache["q"].clone()
        k = self.cache["k"].clone()
        v = self.cache["v"].clone()
        for i in changed:
            q[i] = tokens_new[i] @ self.wq.t() + self.bq
            k[i] = tokens_new[i] @ self.wk.t() + self.bk
            v[i] = tokens_new[i] @ self.wv.t() + self.bv
        s = self.cache["scores"].clone()
        use_dyn = ctx is not None and self.dyn_u is not None and self.dyn_w is not None
        if use_dyn:
            u = ctx @ self.dyn_u.t()
            w = ctx @ self.dyn_w.t()
        else:
            u = self.cache["u"]
            w = self.cache["w"]
            if use_dyn is False and (self.dyn_u is not None):
                pass
        for a in range(t):
            for b in range(t):
                if a in cset or b in cset:
                    val = torch.dot(q[a], k[b]) + self.gab[a, b]
                    if use_dyn:
                        d = torch.clamp(u[a] * w[b], -DYN_CAP, DYN_CAP)
                        val = val + d
                    elif u is not None and w is not None and self.dyn_u is not None:
                        d = torch.clamp(u[a] * w[b], -DYN_CAP, DYN_CAP)
                        val = val + d
                    s[a, b] = val
        g = self.cache["gates"].clone()
        for a in range(t):
            for b in range(t):
                if a in cset or b in cset:
                    g[a, b] = _gate_fn(self.gate, s[a, b])
        y = self.cache["mixed"].clone()
        for a in range(t):
            y[a] = g[a] @ v
        out = self.tokens.clone()
        out = tokens_new + self.alpha * y
        self.tokens = tokens_new.clone()
        self.ctx = None if ctx is None else ctx.clone()
        self.cache = {"q": q, "k": k, "v": v, "scores": s, "gates": g,
                      "mixed": y, "out": out, "u": u, "w": w}
        return out

    def verify_against_full(self, tokens_new, ctx=None, tol=1e-5):
        ref = split_qkv_score_gate_mix(
            tokens_new, self.wq, self.bq, self.wk, self.bk, self.wv, self.bv,
            self.gab, self.gate, self.alpha, ctx, self.dyn_u, self.dyn_w)
        diffs = {}
        for key in ("q", "k", "v", "scores", "gates", "mixed", "out"):
            d = (self.cache[key] - ref[key]).abs().max().item()
            diffs[key] = d
        ok = all(v <= tol for v in diffs.values())
        return ok, diffs
