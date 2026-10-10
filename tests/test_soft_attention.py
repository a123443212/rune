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

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.export.export import export_model, load_exported_arrays, verify_file_hashes
from training.models.rune_models import build_model


def test_soft_spec_and_export_order():
    m = build_model("RUNE-ATTN-SOFT")
    spec = m.model_spec()
    assert spec["arch"] == "RUNE-ATTN-SOFT"
    assert spec["attention"] == "softmax_scaled"
    assert spec["gate"] == "softmax"
    assert spec["arch_version"] == "0.1.0"
    assert sorted(m.arch_tensors().keys()) == sorted(m.export_order())


def test_soft_forward_matches_numpy():
    torch.manual_seed(11)
    m = build_model("RUNE-ATTN-SOFT")
    m.eval()
    ids = [torch.randint(0, 64, (2, 4)) for _ in range(9)]
    masks = [torch.ones(2, 4) for _ in range(9)]
    with torch.no_grad():
        v, w = m(ids, masks)
    assert v.shape == (2,) and w.shape == (2, 3)
    assert bool(((v >= -1.0) & (v <= 1.0)).all())


def test_soft_export_roundtrip(tmp_path):
    torch.manual_seed(12)
    m = build_model("RUNE-ATTN-SOFT")
    m.eval()
    p = str(tmp_path / "soft.rune")
    hdr = export_model(m, p)
    assert hdr["arch"] == "RUNE-ATTN-SOFT"
    assert hdr["attention"] == "softmax_scaled"
    verify_file_hashes(p)
    hdr2, arrays = load_exported_arrays(p)
    assert hdr2["arch"] == "RUNE-ATTN-SOFT"
    assert "wq" in arrays and "gab" in arrays and "w1" in arrays


def test_soft_bucketed_export_order():
    m = build_model("RUNE-ATTN-SOFT", buckets=3)
    order = m.export_order()
    assert "w1_b0" in order and "w1_b2" in order
    assert sorted(m.arch_tensors().keys()) == sorted(order)
