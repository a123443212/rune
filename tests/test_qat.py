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

from training.models.qat import fake_quantize, fake_quantize_model, sym_scales
from training.models.rune_models import build_model


def test_fake_quantize_zeros_identity():
    x = torch.zeros(4, 8)
    assert torch.equal(fake_quantize(x), x)


def test_fake_quantize_bounded_error():
    torch.manual_seed(0)
    x = torch.randn(16, 16)
    q = fake_quantize(x, bits=8)
    assert q.shape == x.shape
    assert bool(((q - x).abs().max() <= x.abs().max() / 127.0 + 1e-6))


def test_sym_scales_positive():
    state = {"w": torch.randn(8, 8), "b": torch.zeros(4)}
    s = sym_scales(state, bits=8)
    assert set(s.keys()) == {"w", "b"}
    assert all(v > 0.0 for v in s.values())


def test_fake_quantize_model_runs_forward():
    torch.manual_seed(1)
    m = build_model("RUNE-MLP")
    m.eval()
    fake_quantize_model(m, bits=8)
    ids = [torch.randint(0, 64, (2, 4)) for _ in range(9)]
    masks = [torch.ones(2, 4) for _ in range(9)]
    with torch.no_grad():
        v, w = m(ids, masks)
    assert v.shape == (2,) and w.shape == (2, 3)
    assert bool(torch.isfinite(v).all())
