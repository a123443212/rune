import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from training.compiler.reference import clip01, dot_tanh, forward_generic, hard_sigmoid, linear_bias_clip, mix_residual, qkv_fused, score_bias_gate


def _rand(*shape, seed=0):
    rng = np.random.RandomState(seed)
    return (rng.randn(*shape).astype(np.float32) * 0.05)


def test_qkv_matches_generic():
    x = _rand(8, 32, seed=1)
    w = _rand(32, 32, seed=2)
    b = _rand(32, seed=3) * 0.1
    q, k, v = qkv_fused(w, b, w, b, w, b, x)
    e = x.dot(w.T) + b
    assert np.abs(q - e).max() < 1e-6


def test_score_bias_gate_parity():
    q = _rand(8, 32, seed=4)
    k = _rand(8, 32, seed=5)
    gab = _rand(8, 8, seed=6) * 0.2
    s, g = score_bias_gate(q, k, gab, False)
    e = q.dot(k.T) + gab
    assert np.abs(s - e).max() < 1e-6
    assert np.abs(g - clip01(e)).max() < 1e-7
    _, gh = score_bias_gate(q, k, gab, True)
    assert np.abs(gh - hard_sigmoid(e)).max() < 1e-7


def test_mix_residual_parity():
    g = clip01(_rand(8, 8, seed=7))
    v = _rand(8, 32, seed=8)
    x = _rand(8, 32, seed=9)
    y = mix_residual(g, v, x, 0.5)
    e = x + 0.5 * g.dot(v)
    assert np.abs(y - e).max() < 1e-6


def test_linear_bias_clip_parity():
    w = _rand(128, 256, seed=10)
    b = _rand(128, seed=11) * 0.1
    vec = _rand(256, seed=12)
    got = linear_bias_clip(w, b, vec)
    e = clip01(w.dot(vec) + b)
    assert np.abs(got - e).max() < 1e-6


def test_full_forward_bounded():
    arrays = {"wq": _rand(32, 32, seed=20), "bq": _rand(32, seed=21) * 0.1, "wk": _rand(32, 32, seed=22), "bk": _rand(32, seed=23) * 0.1, "wvv": _rand(32, 32, seed=24), "bvv": _rand(32, seed=25) * 0.1, "gabS": _rand(8, 8, seed=26) * 0.1, "w1": _rand(128, 256, seed=27), "b1": _rand(128, seed=28) * 0.1, "w2": _rand(32, 128, seed=29), "b2": _rand(32, seed=30) * 0.1, "wvo": _rand(32, seed=31), "bvo": np.array([0.05], dtype=np.float32), "wwdl": _rand(3, 32, seed=32), "bwdl": _rand(3, seed=33) * 0.1}
    flat = clip01(_rand(256, seed=34))
    r = forward_generic(arrays, 8, 32, 128, False, 1.0, flat)
    assert -1.0 <= r["value"] <= 1.0
    assert r["wdl"].shape == (3,)
    assert r["mixed"].shape == (8, 32)


def test_randomized_small_shapes():
    rng = np.random.RandomState(99)
    for trial in range(20):
        m = int(rng.choice([8, 32, 128]))
        n = int(rng.choice([32, 128, 256]))
        mat = (rng.randn(m, n).astype(np.float32) * 0.08)
        vec = (rng.randn(n).astype(np.float32))
        bias = (rng.randn(m).astype(np.float32) * 0.01)
        got = linear_bias_clip(mat, bias, vec)
        e = clip01(mat.dot(vec) + bias)
        assert np.abs(got - e).max() < 1e-5


def test_threshold_boundary():
    def refine(score, thr, thigh, has_low, tlow):
        import math
        if math.isnan(score):
            return True
        if score >= thigh:
            return True
        if has_low and score < tlow:
            return False
        return score >= thr
    thr = 0.5
    for eps in (-1e-7, -1e-6, 0.0, 1e-6, 1e-7):
        s = thr + eps
        assert refine(s, thr, thr, False, 0.0) == (s >= thr)
    assert refine(float("nan"), thr, thr, False, 0.0) is True
    assert refine(0.1, 0.5, 0.8, True, 0.2) is False
