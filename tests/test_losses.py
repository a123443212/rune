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

from training.losses.composite import CompositeLoss
from training.losses.losses import ranking_accuracy, wdl_accuracy


def test_loss_components():
    fn = CompositeLoss(lambda_wdl=0.5, lambda_rank=0.1, rank_margin=0.05, ranking=True)
    v = torch.tensor([0.2, -0.3, 0.9])
    w = torch.randn(3, 3)
    vt = torch.tensor([0.2, -0.3, 0.9])
    wt = torch.tensor([1, 2, 0])
    out = fn(v, w, vt, wt)
    assert out["value"].item() == 0.0
    assert out["rank"].item() == 0.0
    assert out["total"].item() > 0.0


def test_ranking_loss_margin():
    fn = CompositeLoss(lambda_wdl=0.0, lambda_rank=1.0, rank_margin=0.1, ranking=True)
    v = torch.tensor([0.0, 0.0])
    w = torch.zeros(2, 3)
    out = fn(v, w, v, torch.zeros(2, dtype=torch.long),
             rank=(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([1.0])))
    assert abs(out["rank"].item() - 0.1) < 1e-6
    out2 = fn(v, w, v, torch.zeros(2, dtype=torch.long),
              rank=(torch.tensor([0.5]), torch.tensor([0.0]), torch.tensor([1.0])))
    assert out2["rank"].item() == 0.0


def test_ranking_accuracy():
    acc = ranking_accuracy(torch.tensor([0.5, 0.1]), torch.tensor([0.0, 0.4]),
                           torch.tensor([1.0, -1.0]))
    assert acc.item() == 1.0
    acc2 = ranking_accuracy(torch.tensor([0.5, 0.1]), torch.tensor([0.0, 0.4]),
                            torch.tensor([-1.0, 1.0]))
    assert acc2.item() == 0.0


def test_wdl_accuracy():
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    assert wdl_accuracy(logits, torch.tensor([0, 2])).item() == 1.0
