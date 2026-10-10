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

import json
import math
import os
import sys


ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
V10 = os.path.join(ROOT, "spec", "test-vectors", "v10")


def load(name):
    with open(os.path.join(V10, name)) as f:
        return json.load(f)


def clip01(xs):
    out = []
    for x in xs:
        if x != x:
            out.append(x)
            continue
        if x < 0.0:
            out.append(0.0)
        elif x > 1.0:
            out.append(1.0)
        else:
            out.append(x)
    return out


def hard_sigmoid(xs):
    return clip01([0.2 * x + 0.5 for x in xs])


def screlu(xs):
    c = clip01(xs)
    return [v * v for v in c]


def mat_vec(mat, vec, bias, rows, cols):
    out = []
    for r in range(rows):
        a = bias[r]
        base = r * cols
        for k in range(cols):
            a += mat[base + k] * vec[k]
        out.append(a)
    return out


def mat_mul_tt(a, b, m, n, k):
    out = []
    for i in range(m):
        for j in range(n):
            s = 0.0
            for t in range(k):
                s += a[i * k + t] * b[j * k + t]
            out.append(s)
    return out


def mat_mul(a, b, m, n, k):
    out = []
    for i in range(m):
        for j in range(n):
            s = 0.0
            for t in range(k):
                s += a[i * k + t] * b[t * n + j]
            out.append(s)
    return out


def quant_half_away(w, scale, bound):
    if not (scale > 0.0):
        return 0
    if w != w:
        return 0
    q = w / scale
    r = math.floor(q + 0.5) if q >= 0 else math.ceil(q - 0.5)
    if r > bound:
        r = bound
    if r < -bound:
        r = -bound
    if w > 1e30:
        return bound
    if w < -1e30:
        return -bound
    return int(r)


def check_quant():
    data = load("quant.json")
    scale = data["scale"]
    worst = 0
    for w, want in zip(data["inputs"], data["int8"]):
        got = quant_half_away(w, scale, 127)
        worst = max(worst, abs(got - want))
        if got != want:
            print("quant int8 mismatch %r got %r want %r" % (w, got, want))
            return False
    for w, want in zip(data["inputs"], data["int16"]):
        got = quant_half_away(w, scale, 32767)
        worst = max(worst, abs(got - want))
        if got != want:
            print("quant int16 mismatch %r got %r want %r" % (w, got, want))
            return False
    print("quant ok worst=%d" % worst)
    return True


def check_gates():
    cases = [-1.0, -0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0]
    want_clip = [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
    want_hard = [0.3, 0.5, 0.5, 0.55, 0.6, 0.65, 0.7, 0.9]
    want_screlu = [0.0, 0.0, 0.0, 0.0625, 0.25, 0.5625, 1.0, 1.0]
    for got, want in zip(clip01(cases), want_clip):
        if abs(got - want) > 0:
            print("clip mismatch")
            return False
    for got, want in zip(hard_sigmoid(cases), want_hard):
        if abs(got - want) > 1e-6:
            print("hard_sigmoid mismatch")
            return False
    for got, want in zip(screlu(cases), want_screlu):
        if abs(got - want) > 1e-12:
            print("screlu mismatch")
            return False
    print("gates ok")
    return True


def check_mixer():
    data = load("mixer.json")
    t = data["tokens"]
    d = data["dim"]
    x = data["input"]
    q = []
    k = []
    v = []
    for i in range(t):
        xb = x[i * d:(i + 1) * d]
        q += mat_vec(data["wq"], xb, data["bq"], d, d)
        k += mat_vec(data["wk"], xb, data["bk"], d, d)
        v += mat_vec(data["wvv"], xb, data["bvv"], d, d)
    sc = mat_mul_tt(q, k, t, t, d)
    gates = clip01([a + b for a, b in zip(sc, data["gab"])])
    mixed = []
    y = mat_mul(gates, v, t, d, t)
    for i in range(t * d):
        mixed.append(x[i] + y[i])
    worst = 0.0
    for got, want in zip(mixed, data["mixed"]):
        worst = max(worst, abs(got - want))
    print("mixer worst=%.3g" % worst)
    if worst > 1e-5:
        return False
    worst_g = 0.0
    for got, want in zip(gates, data["gates"]):
        worst_g = max(worst_g, abs(got - want))
    print("gates worst=%.3g" % worst_g)
    return worst_g <= 1e-6


def main():
    ok = True
    if not check_quant():
        ok = False
    if not check_gates():
        ok = False
    if not check_mixer():
        ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

