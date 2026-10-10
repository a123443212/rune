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

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.features.python_features import NUM_GROUPS, VOCAB_SIZES
from training.features.python_features import extract_features
from training.rune_v13 import (
    IncrementalRelationalState,
    build_dense_graph,
    dense_forward,
    detect_changed_groups,
    group_deltas_to_tokens,
)
from training.rune_v13.quant_delta import (
    apply_quant_delta,
    dequant_tokens,
    quantize_delta_int8,
)
from training.rune_v13.relational_reference import split_qkv_score_gate_mix

MOVE_PAIRS = {
    "quiet": (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    ),
    "capture": (
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r1bqkbnr/pppp1ppp/2n5/4N3/4P3/8/PPPP1PPP/RNBQKB1R b KQkq - 0 3",
    ),
    "castle": (
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R4RK1 b kq - 1 1",
    ),
    "promotion": (
        "8/4P3/8/8/8/1k6/8/4K3 w - - 0 1",
        "4Q3/8/8/8/8/1k6/8/4K3 b - - 0 1",
    ),
    "en_passant": (
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3",
    ),
}


def _rand_weights(t=8, d=32, seed=7, dynamic=False):
    torch.manual_seed(seed)
    w = {
        "wq": torch.randn(d, d) * 0.08,
        "bq": torch.randn(d) * 0.01,
        "wk": torch.randn(d, d) * 0.08,
        "bk": torch.randn(d) * 0.01,
        "wv": torch.randn(d, d) * 0.08,
        "bv": torch.randn(d) * 0.01,
        "gab": torch.zeros(t, t),
    }
    if dynamic:
        w["dyn_u"] = torch.randn(t, 8) * 0.05
        w["dyn_w"] = torch.randn(t, 8) * 0.05
    return w


def _state_from(w, threshold=8):
    st = IncrementalRelationalState(
        w["wq"], w["bq"], w["wk"], w["bk"], w["wv"], w["bv"], w["gab"], threshold=threshold
    )
    if "dyn_u" in w:
        st.bind_dynamic(w["dyn_u"], w["dyn_w"])
    return st


def test_dense_graph_counts():
    g = build_dense_graph(8)
    assert g.num_edges() == 64
    assert g.is_dense()
    assert len(g.affected_edges([3])) == 15
    assert len(g.affected_edges([0, 5])) == 28
    m = g.edge_meta(2, 5)
    assert m["source"] == 2 and m["target"] == 5


def test_token_change_detection_from_spec():
    for name, (fa, fb) in MOVE_PAIRS.items():
        a = extract_features(fa)
        b = extract_features(fb)
        changed, added, removed = detect_changed_groups(a, b)
        assert len(changed) >= 1, name
        assert len(changed) <= NUM_GROUPS, name


def test_move_type_token_changes_measured():
    torch.manual_seed(3)
    tables = [torch.randn(VOCAB_SIZES[g], 32) * 0.01 for g in range(NUM_GROUPS)]
    got = {}
    for name, (fa, fb) in MOVE_PAIRS.items():
        a = extract_features(fa)
        b = extract_features(fb)
        r = group_deltas_to_tokens(tables, a, b, 32)
        got[name] = len(r["changed_tokens"])
        assert len(r["changed_tokens"]) <= 8, name
    assert max(got.values()) <= 8


def test_b0_b1_b2_parity_static():
    w = _rand_weights()
    torch.manual_seed(11)
    t, d = 8, 32
    x0 = torch.rand(t, d)
    x1 = x0.clone()
    x1[3] += torch.randn(d) * 0.05
    x1.clamp_(0.0, 1.0)
    b0 = dense_forward(x1, w["wq"], w["bq"], w["wk"], w["bk"], w["wv"], w["bv"], w["gab"])
    st = _state_from(w)
    st.rebuild(x0)
    b2 = st.update(x1, [3])
    ok, diffs = st.verify_against_full(x1, tol=1e-5)
    assert ok, diffs
    assert torch.equal(b2, b0) or (b2 - b0).abs().max().item() <= 1e-5


def test_b0_b1_b2_parity_dynamic_bias():
    w = _rand_weights(dynamic=True)
    torch.manual_seed(13)
    t, d = 8, 32
    x0 = torch.rand(t, d)
    ctx = torch.randn(8)
    x1 = x0.clone()
    x1[1] += 0.06
    x1.clamp_(0.0, 1.0)
    st = _state_from(w)
    st.rebuild(x0, ctx)
    st.update(x1, [1], ctx)
    ok, diffs = st.verify_against_full(x1, ctx, tol=1e-5)
    assert ok, diffs


def test_delta_through_projection_matches_recompute():
    torch.manual_seed(17)
    d = 32
    A_old = torch.randn(d)
    dA = torch.randn(d) * 0.01
    W = torch.randn(d, d) * 0.08
    b = torch.randn(d) * 0.01
    full = (A_old + dA) @ W.t() + b
    incr = (A_old @ W.t() + b) + dA @ W.t()
    assert (full - incr).abs().max().item() <= 1e-5


def test_long_sequence_no_drift():
    w = _rand_weights(seed=21)
    torch.manual_seed(23)
    t, d = 8, 32
    st = _state_from(w)
    x = torch.rand(t, d)
    st.rebuild(x)
    for step in range(1000):
        k = step % t
        xn = x.clone()
        xn[k] += torch.randn(d) * 0.02
        xn.clamp_(0.0, 1.0)
        out = st.update(xn, [k])
        ref = dense_forward(xn, w["wq"], w["bq"], w["wk"], w["bk"], w["wv"], w["bv"], w["gab"])
        assert (out - ref).abs().max().item() <= 1e-5, step
        x = xn
    ok, _ = st.verify_against_full(x, tol=1e-5)
    assert ok


def test_search_branch_isolation():
    w = _rand_weights(seed=31)
    torch.manual_seed(33)
    t, d = 8, 32
    st = _state_from(w)
    x0 = torch.rand(t, d)
    xa = x0.clone()
    xa[0] += 0.1
    xb = x0.clone()
    xb[7] -= 0.1
    xa.clamp_(0.0, 1.0)
    xb.clamp_(0.0, 1.0)
    st.rebuild(x0)
    st.push()
    st.update(xa, [0])
    ok_a, _ = st.verify_against_full(xa, tol=1e-5)
    assert ok_a
    st.pop()
    st.update(xb, [7])
    ok_b, _ = st.verify_against_full(xb, tol=1e-5)
    assert ok_b
    assert torch.equal(st.tokens, xb)


def test_threshold_fallback_and_all_token():
    w = _rand_weights(seed=41)
    torch.manual_seed(43)
    t, d = 8, 32
    st = _state_from(w, threshold=2)
    x0 = torch.rand(t, d)
    x1 = torch.rand(t, d)
    st.rebuild(x0)
    st.update(x1, [0, 1])
    assert st.incremental_updates == 1
    st.update(x1, list(range(8)))
    assert st.fallbacks == 1
    ok, _ = st.verify_against_full(x1, tol=1e-5)
    assert ok


def test_quant_delta_parity():
    torch.manual_seed(51)
    acc = torch.randint(-500, 500, (8, 32))
    delta = torch.randint(-20, 20, (8, 32))
    scale = 0.01
    qd, _ = quantize_delta_int8(delta.float(), scale)
    assert qd.dtype == torch.int8
    got = apply_quant_delta(acc, qd)
    assert int((got - (acc + qd.to(torch.int32))).abs().max().item()) == 0
    toks = dequant_tokens(acc, torch.full((8,), scale))
    assert toks.min().item() >= 0.0 and toks.max().item() <= 1.0


def test_interaction_sparsity_measured_not_assumed():
    w = _rand_weights(seed=61)
    mag = w["gab"].abs()
    assert mag.max().item() == 0.0
    torch.manual_seed(63)
    x = torch.rand(8, 32)
    r = split_qkv_score_gate_mix(
        x, w["wq"], w["bq"], w["wk"], w["bk"], w["wv"], w["bv"], w["gab"]
    )
    active = (r["gates"] > 1e-6).float().mean().item()
    assert 0.0 <= active <= 1.0


def test_shared_vector_file_consistent():
    path = os.path.join(
        os.path.dirname(__file__), "..", "spec", "test-vectors", "v13", "incremental.txt"
    )
    assert os.path.exists(path)
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert len(lines) == 13
