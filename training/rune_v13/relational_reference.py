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

DYN_CAP = 0.25


def _gate(name, s):
    if name == "hard_sigmoid":
        return torch.clamp(0.2 * s + 0.5, 0.0, 1.0)
    return torch.clamp(s, 0.0, 1.0)


def split_qkv_score_gate_mix(x, wq, bq, wk, bk, wv, bv, gab, gate="clip", alpha=1.0,
                             ctx=None, dyn_u=None, dyn_w=None):
    q = x @ wq.t() + bq
    k = x @ wk.t() + bk
    v = x @ wv.t() + bv
    s = q @ k.t() + gab
    u = None
    w = None
    if ctx is not None and dyn_u is not None and dyn_w is not None:
        u = ctx @ dyn_u.t()
        w = ctx @ dyn_w.t()
        s = s + torch.clamp(u.unsqueeze(-1) * w.unsqueeze(-2), -DYN_CAP, DYN_CAP)
    g = _gate(gate, s)
    y = g @ v
    out = x + alpha * y
    return {"q": q, "k": k, "v": v, "scores": s, "gates": g, "mixed": y, "out": out,
            "u": u, "w": w}


def dense_forward(x, wq, bq, wk, bk, wv, bv, gab, gate="clip", alpha=1.0,
                  ctx=None, dyn_u=None, dyn_w=None):
    return split_qkv_score_gate_mix(x, wq, bq, wk, bk, wv, bv, gab, gate, alpha,
                                    ctx, dyn_u, dyn_w)["out"]
