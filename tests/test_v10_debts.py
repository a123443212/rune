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
import subprocess
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import torch
ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS = os.path.join(ROOT, "spec", "test-vectors", "models")
CPP_EVAL = os.path.join(ROOT, "build", "rune_eval")
RUST_BIN = os.path.join(ROOT, "target", "release", "rune")
FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
]
needs_cpp = pytest.mark.skipif(not os.path.exists(CPP_EVAL), reason="build/rune_eval missing")
needs_rust = pytest.mark.skipif(not os.path.exists(RUST_BIN), reason="target/release/rune missing")
def dense_torch(path):
    sys.path.insert(0, os.path.join(ROOT, "tools", "diff"))
    import importlib
    dc = importlib.import_module("dense_check")
    return dc.load_torch_dense(path)
def test_dense_int8_close_to_fp32_torch():
    sys.path.insert(0, os.path.join(ROOT, "tools", "diff"))
    import importlib
    dc = importlib.import_module("dense_check")
    mf = dc.load_torch_dense(os.path.join(MODELS, "dense-b-fp32.rune"))
    m8 = dc.load_torch_dense(os.path.join(MODELS, "dense-b-int8.rune"))
    m16 = dc.load_torch_dense(os.path.join(MODELS, "dense-b-int16.rune"))
    with torch.no_grad():
        for fen in FENS:
            ids, masks = dc.batch_for(fen)
            vf, _ = mf(ids, masks)
            v8, _ = m8(ids, masks)
            v16, _ = m16(ids, masks)
            assert abs(float(vf.item()) - float(v8.item())) < 0.01, fen
            assert abs(float(vf.item()) - float(v16.item())) < 0.005, fen
@needs_cpp
def test_dense_py_cpp_via_tool(tmp_path):
    pos = tmp_path / "pos.txt"
    pos.write_text("\n".join(FENS) + "\n")
    out = tmp_path / "rep.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "diff", "dense_check.py"),
                        "--models", os.path.join(MODELS, "dense-b-fp32.rune"),
                        os.path.join(MODELS, "dense-b-int8.rune"),
                        "--positions", str(pos), "--cpp-bin", CPP_EVAL,
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
@needs_cpp
def test_adaptive_sweep_via_tool(tmp_path):
    pos = tmp_path / "pos.txt"
    pos.write_text("\n".join(FENS) + "\n")
    out = tmp_path / "rep.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "bench", "adaptive_sweep.py"),
                        "--model", os.path.join(MODELS, "adaptive-fp32.rune"),
                        "--positions", str(pos), "--cpp-bin", CPP_EVAL,
                        "--thresholds", "0.0,0.06,0.07,0.5",
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
@needs_cpp
@needs_rust
def test_rel_live_gate_triangle(tmp_path):
    pos = tmp_path / "pos.txt"
    pos.write_text("1n2k2r/Nb1pb3/1r1n3p/4p1qP/pPP1pbPR/8/P7/R5K1 w k - 0 31\n")
    out = tmp_path / "rep.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "diff", "cross_check.py"),
                        "--model", os.path.join(MODELS, "rel-08x32-fp32.rune"),
                        "--positions", str(pos), "--cpp-bin", CPP_EVAL,
                        "--rust-bin", RUST_BIN, "--tol", "2e-5",
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
@needs_cpp
@needs_rust
def test_mixed_path_triangle_simd(tmp_path):
    pos = tmp_path / "pos.txt"
    pos.write_text("\n".join(FENS) + "\n")
    out = tmp_path / "rep.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "diff", "cross_check.py"),
                        "--model", os.path.join(MODELS, "small-gab-fp32.rune"),
                        "--positions", str(pos), "--cpp-bin", CPP_EVAL,
                        "--cpp-path", "scalar", "--rust-bin", RUST_BIN,
                        "--rust-kernel", "simd", "--tol", "2e-5",
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
