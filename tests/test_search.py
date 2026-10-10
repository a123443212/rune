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

from training.losses.uncertainty import SearchLoss
from training.models.uncertainty import (
    ARCH_ID,
    ARCH_VERSION,
    ROUTING_MODES,
    build_search_model,
    route_multi,
)


def tiny_batch(n=4):
    ids = [torch.randint(0, 64, (n, 4)) for _ in range(9)]
    masks = [torch.ones(n, 4) for _ in range(9)]
    return ids, masks


def test_search_forward_shapes_and_bounds():
    m = build_search_model(dim=16, stability_on=True)
    m.eval()
    ids, masks = tiny_batch()
    with torch.no_grad():
        cv, cw, rv, rw, diff, u, s = m(ids, masks)
    assert cv.shape == (4,) and rv.shape == (4,)
    assert u.shape == (4,) and s.shape == (4,)
    assert bool(((u >= 0.0) & (u <= 1.0)).all())
    assert bool((s >= 0.0).all())


def test_infer_search_modes_deterministic():
    torch.manual_seed(0)
    m = build_search_model(dim=16)
    m.eval()
    ids, masks = tiny_batch()
    t = {"difficulty": -1e9, "uncertainty": 0.5, "stability": 0.5}
    with torch.no_grad():
        _, _, mall, ua, _ = m.infer_search(ids, masks, "full", t)
        _, _, mall2, ub, _ = m.infer_search(ids, masks, "full", t)
        _, _, mdiff, _, _ = m.infer_search(ids, masks, "difficulty",
                                          {"difficulty": 1e9, "uncertainty": 0.5,
                                           "stability": 0.5})
    assert mall.all()
    assert torch.equal(mall, mall2)
    assert torch.equal(ua, ub)
    assert not mdiff.any()
    with pytest.raises(ValueError):
        m.infer_search(ids, masks, "bogus", t)


def test_route_multi_semantics():
    d = torch.tensor([0.2, 0.8])
    u = torch.tensor([0.9, 0.1])
    s = torch.tensor([0.0, 0.9])
    t = {"difficulty": 0.5, "uncertainty": 0.5, "stability": 0.5}
    sig = {"difficulty": d, "uncertainty": u, "stability": s}
    assert route_multi(sig, "difficulty", t).tolist() == [False, True]
    assert route_multi(sig, "uncertainty", t).tolist() == [True, False]
    assert route_multi(sig, "both", t).tolist() == [True, True]
    assert route_multi(sig, "full", t).tolist() == [True, True]


def test_search_loss_gating():
    loss_on = SearchLoss(uncertainty=True, stability=False)
    loss_off = SearchLoss(uncertainty=False, stability=False)
    n = 4
    zeros_w = torch.zeros(n, 3)
    args = (torch.zeros(n), zeros_w, torch.zeros(n), zeros_w,
            torch.zeros(n), torch.sigmoid(torch.zeros(n)), torch.zeros(n),
            torch.zeros(n), torch.zeros(n, dtype=torch.long))
    parts_on = loss_on(*args)
    parts_off = loss_off(*args)
    assert parts_on["unc"].item() >= 0.0
    assert parts_off["unc"].item() == 0.0
    assert parts_on["stab"].item() == 0.0
    assert float(parts_on["stab_skipped"].item()) == 1.0
    tgt = torch.ones(n)
    msk = torch.ones(n)
    parts_stab = SearchLoss(stability=True)(*args[:9], rank=None, oracle=None,
                                            s_target=tgt, s_mask=msk)
    assert parts_stab["stab"].item() > 0.0


def test_search_spec_and_export_order():
    m = build_search_model(dim=16)
    s = m.model_spec()
    assert s["arch"] == ARCH_ID and s["arch_version"] == ARCH_VERSION
    assert s["uncertainty"] is True and s["stability_head"] is False
    assert m.export_order()[-4:] == ["uw", "ub", "sw", "sb"]
    assert set(m.export_order()) == set(m.arch_tensors().keys())
    assert "difficulty" in ROUTING_MODES and "full" in ROUTING_MODES


def test_search_export_header(tmp_path):
    import struct

    import numpy as np

    from training.export.export import export_model, fnv1a, load_exported_arrays

    torch.manual_seed(0)
    m = build_search_model(dim=16)
    m.eval()
    path = str(tmp_path / "r05.rune")
    header = export_model(m, path, quantization="fp32")
    assert header["arch"] == "RUNE-05"
    assert header["uncertainty"] is True
    assert len(header["checksum"]) == 16
    with open(path, "rb") as f:
        assert f.read(4) == b"RUNE"
        (nbytes,) = struct.unpack("<I", f.read(4))
        f.read(nbytes)
        payload = f.read()
    assert format(fnv1a(payload), "016x") == header["checksum"]
    _, arrays = load_exported_arrays(path)
    assert "uw" in arrays and "sb" in arrays
    assert int(np.asarray(arrays["uw"]).size) == 16 * 8
