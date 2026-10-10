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

import numpy as np


def clip01(x):
    return np.minimum(np.maximum(x, 0.0), 1.0)


def hard_sigmoid(s):
    return np.minimum(np.maximum(0.2 * s + 0.5, 0.0), 1.0)


def screlu(s):
    c = np.minimum(np.maximum(s, 0.0), 1.0)
    return c * c


def matvec_generic(mat, vec, bias):
    return mat.dot(vec) + bias


def matvec_32(mat, vec, bias):
    acc = bias.copy()
    for r in range(32):
        s = bias[r]
        row = mat[r]
        for c in range(32):
            s += row[c] * vec[c]
        acc[r] = s
    return acc


def qkv_fused(wq, bq, wk, bk, wv, bv, x):
    t = x.shape[0]
    q = np.zeros_like(x)
    k = np.zeros_like(x)
    v = np.zeros_like(x)
    for i in range(t):
        q[i] = wq.dot(x[i]) + bq
        k[i] = wk.dot(x[i]) + bk
        v[i] = wv.dot(x[i]) + bv
    return q, k, v


def score_bias_gate(q, k, gab, gate="clip"):
    s = q.dot(k.T) + gab
    if gate == "hard_sigmoid":
        g = hard_sigmoid(s)
    elif gate == "screlu":
        g = screlu(s)
    elif gate == "clip":
        g = clip01(s)
    else:
        raise ValueError(f"unknown gate {gate}")
    return s, g


def mix_residual(g, v, x, alpha):
    y = g.dot(v)
    return x + alpha * y


def linear_bias_clip(w, b, vec):
    return clip01(w.dot(vec) + b)


def dot_tanh(wvo, bvo, h2):
    import math
    w = np.asarray(wvo, dtype=np.float64).reshape(-1)
    h = np.asarray(h2, dtype=np.float64).reshape(-1)
    return math.tanh(float(np.dot(w, h) + float(np.asarray(bvo).reshape(-1)[0])))


def forward_generic(arrays, tokens, dim, h1n, gate, alpha, flat):
    x = flat.reshape(tokens, dim)
    wq = arrays["wq"].reshape(dim, dim)
    bq = arrays["bq"]
    wk = arrays["wk"].reshape(dim, dim)
    bk = arrays["bk"]
    wv = arrays["wvv"].reshape(dim, dim) if "wvv" in arrays else arrays["wv"].reshape(dim, dim)
    bv = arrays["bvv"] if "bvv" in arrays else arrays["bv"]
    gab = arrays["gabS"].reshape(tokens, tokens) if "gabS" in arrays else arrays["gab"].reshape(tokens, tokens)
    q, k, v = qkv_fused(wq, bq, wk, bk, wv, bv, x)
    s, g = score_bias_gate(q, k, gab, gate)
    mixed = mix_residual(g, v, x, alpha)
    flat2 = mixed.reshape(-1)
    w1 = arrays["w1"].reshape(h1n, tokens * dim)
    b1 = arrays["b1"]
    w2raw = np.asarray(arrays["w2"]).reshape(-1)
    if w2raw.size == 32 * h1n * 2:
        raise ValueError("pair head (w2 [32,2*H1]) not yet supported in compiled path; use direct eval (rune-cli eval / match) for head_pair models")
    w2 = arrays["w2"].reshape(32, h1n)
    b2 = arrays["b2"]
    h1 = linear_bias_clip(w1, b1, flat2)
    h2 = linear_bias_clip(w2, b2, h1)
    val = dot_tanh(arrays["wvo"], arrays["bvo"], h2)
    wwdl = arrays["wwdl"].reshape(3, 32)
    bwdl = arrays["bwdl"]
    wdl = wwdl.dot(h2) + bwdl
    return {"q": q, "k": k, "v": v, "scores": s, "gate": g, "mixed": mixed, "h1": h1, "h2": h2, "value": val, "wdl": wdl}


def forward_compiled(arrays, tokens, dim, h1n, gate, alpha, flat):
    return forward_generic(arrays, tokens, dim, h1n, gate, alpha, flat)
