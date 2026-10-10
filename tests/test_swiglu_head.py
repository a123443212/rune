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

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.export.export import export_model, verify_file_hashes
from training.models.rune_models import build_model
from training.models.relational import build_rel_model


def test_swiglu_spec_and_tensors():
    m = build_model("RUNE-MLP", head="value_swiglu")
    spec = m.model_spec()
    assert spec["head"] == "value_swiglu"
    assert spec["arch_version"] == "0.3.0"
    keys = m.arch_tensors().keys()
    assert "wgate" in keys and "wup" in keys and "w1" not in keys
    assert sorted(keys) == sorted(m.export_order())


def test_swiglu_rejects_pair():
    with pytest.raises(ValueError):
        build_model("RUNE-MLP", head="value_swiglu", pair=True)
    with pytest.raises(ValueError):
        build_rel_model(head="value_swiglu", pair=True)


def test_swiglu_forward_shapes():
    m = build_model("RUNE-ATTN-GAB", head="value_swiglu")
    m.eval()
    ids = [torch.randint(0, 64, (2, 4)) for _ in range(9)]
    masks = [torch.ones(2, 4) for _ in range(9)]
    with torch.no_grad():
        v, w = m(ids, masks)
    assert v.shape == (2,) and w.shape == (2, 3)


def test_swiglu_relational_export(tmp_path):
    torch.manual_seed(7)
    r = build_rel_model(head="value_swiglu")
    r.eval()
    assert r.model_spec()["head"] == "value_swiglu"
    assert "wgate" in r.arch_tensors()
    assert "wgate" in r.export_order()


def test_swiglu_export_roundtrip(tmp_path):
    torch.manual_seed(9)
    m = build_model("RUNE-MLP", head="value_swiglu")
    m.eval()
    p = str(tmp_path / "swiglu.rune")
    hdr = export_model(m, p)
    assert hdr["head"] == "value_swiglu"
    assert hdr["arch_version"] == "0.3.0"
    verify_file_hashes(p)
