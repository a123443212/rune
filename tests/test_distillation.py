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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from training.losses.distillation import (
    DistillCompositeLoss,
    WeightedDistillationLoss,
    confidence_weights,
)


def test_confidence_weights_bounded():
    u = torch.tensor([0.0, 0.5, 1.0])
    for mode in ("uniform", "confidence", "difficulty"):
        w = confidence_weights(u, mode)
        assert bool(((w >= 0.25) & (w <= 1.0)).all())
    assert torch.equal(confidence_weights(u, "uniform"), torch.ones(3))
    assert confidence_weights(u, "confidence")[0] > confidence_weights(u, "confidence")[2]
    assert confidence_weights(u, "difficulty")[2] > confidence_weights(u, "difficulty")[0]
    with pytest.raises(ValueError):
        confidence_weights(u, "bogus")


def test_weighted_distill_matches_uniform():
    torch.manual_seed(0)
    s = torch.randn(8)
    t = torch.randn(8)
    plain = WeightedDistillationLoss()(s, t, torch.ones(8))
    import torch.nn.functional as F

    assert abs(plain.item() - F.mse_loss(s, t).item()) < 1e-6


def test_distill_composite_task_and_alpha():
    torch.manual_seed(1)
    n = 6
    sv, tv = torch.randn(n, requires_grad=True), torch.randn(n)
    sw = torch.randn(n, 3, requires_grad=True)
    tw = torch.randn(n, 3)
    value = torch.randn(n)
    wdl = torch.randint(0, 3, (n,))
    fn = DistillCompositeLoss(task="value_wdl", alpha=0.5)
    parts = fn(sv, sw, tv, tw, value, wdl)
    assert parts["task_value"].item() > 0
    assert parts["task_wdl"].item() > 0
    assert parts["distill"].item() > 0
    assert parts["distill_wdl"].item() > 0
    assert parts["rank"].item() == 0.0
    pure = DistillCompositeLoss(task="value_wdl", alpha=0.0)
    assert pure(sv, sw, tv, tw, value, wdl)["distill"].item() > 0
    nodist = DistillCompositeLoss(task="value_wdl", alpha=0.5)
    p0 = nodist(sv, sw, None, None, value, wdl)
    assert p0["distill"].item() == 0.0
    assert p0["distill_wmean"].item() == 0.0


def test_distill_weighted_vs_uniform_control():
    torch.manual_seed(2)
    n = 8
    sv = torch.randn(n)
    tv = torch.randn(n)
    value = torch.randn(n)
    wdl = torch.randint(0, 3, (n,))
    sw = torch.randn(n, 3)
    tu = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    uni = DistillCompositeLoss(weight_mode="uniform")(sv, sw, tv, None, value, wdl)
    con = DistillCompositeLoss(weight_mode="confidence")(
        sv, sw, tv, None, value, wdl, teacher_u=tu)
    assert uni["distill_wmean"].item() == 1.0
    assert 0.25 <= con["distill_wmean"].item() <= 1.0
    assert con["distill"].item() != uni["distill"].item()
