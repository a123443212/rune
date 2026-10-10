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
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.models.rune_models import ARCH_IDS, EXPORT_ORDER, build_model
from training.experiments import screening
from training.experiments.model_registry import MODEL_IDS, pair_enabled_for_model
from training.trainer.trainer import Trainer


def test_all_archs_forward_shapes():
    for arch in ARCH_IDS:
        model = build_model(arch)
        model.eval()
        ids = [torch.randint(0, 8, (4, 6)) for _ in range(9)]
        masks = [torch.ones(4, 6) for _ in range(9)]
        with torch.no_grad():
            v, w = model(ids, masks)
        assert v.shape == (4,)
        assert w.shape == (4, 3)
        assert bool(((v >= -1.0) & (v <= 1.0)).all())


def test_parameter_accounting():
    counts = {}
    for arch in ARCH_IDS:
        model = build_model(arch)
        counts[arch] = model.parameter_count()
        assert model.model_size_bytes() == counts[arch] * 4
    assert counts["RUNE-SFNN"] > counts["RUNE-MLP"]
    assert counts["RUNE-ATTN"] > counts["RUNE-MLP"]
    assert counts["RUNE-ATTN"] == counts["RUNE-ATTN-GAB"]
    mlp_tensors = sum(v.numel() for v in build_model("RUNE-MLP").arch_tensors().values())
    assert mlp_tensors == 37156


def test_attention_equations_numpy():
    torch.manual_seed(0)
    model = build_model("RUNE-ATTN-GAB")
    model.eval()
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        y = model.attn(x)
    xn = x.numpy()
    wq = model.attn.wq.weight.detach().numpy()
    bq = model.attn.wq.bias.detach().numpy()
    wk = model.attn.wk.weight.detach().numpy()
    bk = model.attn.wk.bias.detach().numpy()
    wv = model.attn.wv.weight.detach().numpy()
    bv = model.attn.wv.bias.detach().numpy()
    gab = model.attn.gab.detach().numpy()
    for b in range(2):
        q = xn[b] @ wq.T + bq
        k = xn[b] @ wk.T + bk
        v = xn[b] @ wv.T + bv
        s = q @ k.T + gab
        a = np.clip(s, 0, 1)
        expect = xn[b] + a @ v
        assert np.allclose(y[b].numpy(), expect, atol=1e-5)


def test_gab_ablation_switch():
    torch.manual_seed(1)
    a = build_model("RUNE-ATTN")
    g = build_model("RUNE-ATTN-GAB")
    g.attn.load_state_dict(a.attn.state_dict(), strict=False)
    x = torch.randn(3, 8, 32)
    with torch.no_grad():
        ya = a.attn(x)
        yg = g.attn(x)
    assert torch.allclose(ya, yg, atol=1e-6)
    with torch.no_grad():
        g.attn.gab.fill_(0.5)
        yg2 = g.attn(x)
    assert not torch.allclose(ya, yg2, atol=1e-4)


def test_spec_and_export_order_cover_all_params():
    for arch in ARCH_IDS:
        model = build_model(arch)
        spec = model.model_spec()
        assert spec["tokens"] == 8 and spec["token_dim"] == 32
        arch_t = model.arch_tensors()
        assert sorted(arch_t.keys()) == sorted(model.export_order())
        emb = model.embedding_tensors()
        assert len(emb) == 9


def test_pair_candidate_is_registered_and_trained_as_pair(monkeypatch):
    config = {
        "models": ["rune_attn_gab", "rune_attn_gab_pair"],
        "research": {"allow_experimental": True},
        "architecture": {"pair": True},
        "pair_legs": {
            "control": {"model": "rune_attn_gab", "pair": False},
            "candidate": {"model": "rune_attn_gab_pair", "pair": True},
        },
    }
    assert MODEL_IDS["rune_attn_gab_pair"] == "RUNE-ATTN-GAB"
    assert pair_enabled_for_model(config, "rune_attn_gab") is False
    assert pair_enabled_for_model(config, "rune_attn_gab_pair") is True
    trainer_configs = []
    monkeypatch.setattr(screening, "Trainer", lambda trainer_config: trainer_configs.append(trainer_config))
    runner = screening.ScreeningRunner(config)
    runner.run_model("rune_attn_gab", {"val": []}, {}, [])
    runner.run_model("rune_attn_gab_pair", {"val": []}, {}, [])
    assert [item["pair"] for item in trainer_configs] == [False, True]
    trainer = Trainer({"arch": MODEL_IDS["rune_attn_gab_pair"], "pair": True})
    assert trainer.model.model_spec()["arch_version"] == "0.2.0"
    assert trainer.model.head.fc2.in_features == 256
