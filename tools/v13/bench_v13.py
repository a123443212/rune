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

import sys
import time

import torch

sys.path.insert(0, ".")

from training.rune_v13 import IncrementalRelationalState
from training.rune_v13.interaction_graph import build_dense_graph
from training.rune_v13.relational_reference import dense_forward

T = 8
D = 32
REPEATS = 200


def make_weights(seed=7):
    torch.manual_seed(seed)
    return {
        "wq": torch.randn(D, D) * 0.08,
        "bq": torch.randn(D) * 0.01,
        "wk": torch.randn(D, D) * 0.08,
        "bk": torch.randn(D) * 0.01,
        "wv": torch.randn(D, D) * 0.08,
        "bv": torch.randn(D) * 0.01,
        "gab": torch.zeros(T, T),
    }


def bench_fn(fn, repeats=REPEATS):
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / repeats * 1e9


def main():
    w = make_weights()
    torch.manual_seed(99)
    x0 = torch.rand(T, D)
    print("changed,b0_ns,b2_ns,cells_recomputed,cells_total")
    for k in (1, 2, 3, 4, 8):
        changed = list(range(k))
        torch.manual_seed(100 + k)
        delta = torch.randn(T, D) * 0.03
        delta[k:] = 0.0
        x1 = (x0 + delta).clamp(0.0, 1.0)
        b0 = bench_fn(lambda: dense_forward(x1, w["wq"], w["bq"], w["wk"], w["bk"], w["wv"], w["bv"], w["gab"]))
        st = IncrementalRelationalState(
            w["wq"], w["bq"], w["wk"], w["bk"], w["wv"], w["bv"], w["gab"], threshold=8
        )
        st.rebuild(x0)
        b2 = bench_fn(lambda: st.update(x1, changed))
        cells = 2 * k * T - k * k
        print(f"{k},{b0:.0f},{b2:.0f},{cells},{T * T}")
    print("threshold,fallbacks,incr_updates")
    for thr in (1, 2, 3, 4):
        st = IncrementalRelationalState(
            w["wq"], w["bq"], w["wk"], w["bk"], w["wv"], w["bv"], w["gab"], threshold=thr
        )
        st.rebuild(x0)
        torch.manual_seed(5)
        for step in range(20):
            kk = (step % 4) + 1
            xn = (x0 + torch.randn(T, D) * 0.02).clamp(0.0, 1.0)
            st.update(xn, list(range(kk)))
        print(f"{thr},{st.fallbacks},{st.incremental_updates}")
    g = build_dense_graph(T)
    print(f"graph_version={g.version} dense_edges={g.num_edges()}")


if __name__ == "__main__":
    main()
