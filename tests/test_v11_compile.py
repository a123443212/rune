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
import os
import struct
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from training.compiler.artifact import cache_key, read_compiled_header, write_compiled
from training.compiler.graph import export_canonical_graph, load_canonical_graph
from training.compiler.ir import build_ir, verify_ir
from training.compiler.memory import plan_memory
from training.compiler.packing import PACKING_VERSION, pack_weights
from training.compiler.plan import select_kernels
from training.compiler.precomputed import build_precomputed


def _mini_header(tmp):
    h = {"format": 2, "architecture_id": "RUNE-ATTN-GAB", "architecture_version": "0.2.0", "feature_version": "grouped_hkav2_fullthreats_v02", "tokens": 8, "token_dim": 32, "quantization": "fp32", "model_hash": "00"}
    return h


def test_packing_deterministic(tmp_path):
    rng = np.random.RandomState(7)
    a = rng.randn(32, 32).astype(np.float32) * 0.05
    arrays = {"wq": a}
    metas = [{"name": "wq", "shape": [32, 32], "dtype": "float32"}]
    p1, m1 = pack_weights(arrays, metas, "avx2")
    p2, m2 = pack_weights(arrays, metas, "avx2")
    assert np.array_equal(p1["wq"], p2["wq"])
    assert m1 == m2
    assert m1["packing_version"] == PACKING_VERSION


def test_artifact_reproducible(tmp_path):
    rng = np.random.RandomState(11)
    arrays = {"w1": rng.randn(4, 8).astype(np.float32), "b1": np.zeros((4,), dtype=np.float32)}
    order = ["w1", "b1"]
    header = _mini_header(tmp_path)
    spec = {"arch": "RUNE-ATTN-GAB", "tokens": 8, "token_dim": 32, "quantization": "fp32", "gate": "clip", "alpha": 1.0, "head_h1": 4, "head_h2": 2}
    ir = build_ir(spec, [{"name": "w1", "shape": [4, 8], "dtype": "float32"}, {"name": "b1", "shape": [4], "dtype": "float32"}], {"cpu": "generic-x86-64", "isa": "portable", "vector_width": 1})
    ir["kernel_plan"], ir["fusion"] = select_kernels(ir)
    ir["memory"] = plan_memory(ir)
    pre = {"packing_meta": {"packing_version": 1}, "constants": build_precomputed(spec, arrays, {})}
    o1 = str(tmp_path / "a.rune")
    o2 = str(tmp_path / "b.rune")
    i1 = write_compiled(o1, header, arrays, order, ir, pre)
    i2 = write_compiled(o2, header, arrays, order, ir, pre)
    with open(o1, "rb") as f:
        b1 = f.read()
    with open(o2, "rb") as f:
        b2 = f.read()
    assert b1 == b2
    assert i1["source_hash"] == i2["source_hash"]
    assert i1["plan_hash"] == i2["plan_hash"]


def test_compiled_header_checks(tmp_path):
    rng = np.random.RandomState(3)
    arrays = {"w1": rng.randn(4, 8).astype(np.float32), "b1": np.zeros((4,), dtype=np.float32)}
    order = ["w1", "b1"]
    header = _mini_header(tmp_path)
    spec = {"arch": "RUNE-ATTN-GAB", "tokens": 8, "token_dim": 32, "quantization": "fp32", "gate": "clip", "alpha": 1.0, "head_h1": 4, "head_h2": 2}
    ir = build_ir(spec, [{"name": "w1", "shape": [4, 8], "dtype": "float32"}, {"name": "b1", "shape": [4], "dtype": "float32"}], {"cpu": "generic-x86-64", "isa": "avx2", "vector_width": 8})
    ir["kernel_plan"], ir["fusion"] = select_kernels(ir)
    ir["memory"] = plan_memory(ir)
    pre = {"packing_meta": {}, "constants": build_precomputed(spec, arrays, {})}
    out = str(tmp_path / "c.rune")
    write_compiled(out, header, arrays, order, ir, pre)
    h = read_compiled_header(out)
    assert h["compiled"] is True
    assert h["target_isa"] == "avx2"
    assert h["rune_ir_version"] == "1.1"
    assert "kernel_plan" in h
    assert "memory_plan" in h
    assert "source_hash" in h


def test_cache_key_stable():
    k1 = cache_key("abcd", "RUNE-ATTN-GAB", "fp32", "avx2")
    k2 = cache_key("abcd", "RUNE-ATTN-GAB", "fp32", "avx2")
    k3 = cache_key("abcd", "RUNE-ATTN-GAB", "fp32", "portable")
    assert k1 == k2
    assert k1 != k3


def test_graph_roundtrip(tmp_path):
    spec = {"arch": "RUNE-ATTN-GAB", "tokens": 8, "token_dim": 32, "quantization": "fp32", "gate": "clip", "alpha": 1.0, "head_h1": 128, "head_h2": 32}
    ir = build_ir(spec, [], {"cpu": "generic-x86-64", "isa": "portable", "vector_width": 1})
    ir["kernel_plan"], ir["fusion"] = select_kernels(ir)
    ir["memory"] = plan_memory(ir)
    p = str(tmp_path / "g.json")
    export_canonical_graph(ir, p)
    back = load_canonical_graph(p)
    assert back["model"]["architecture"] == "RUNE-ATTN-GAB"
    assert len(back["ops"]) == 15
