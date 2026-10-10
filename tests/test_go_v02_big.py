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

import torch


def resnet_dir():
    return os.path.join(os.path.dirname(__file__), "..", "spec", "test-vectors", "resnet")


def models_dir():
    return os.path.join(os.path.dirname(__file__), "..", "spec", "test-vectors", "models")


def load_fixture_model(n, blocks):
    from training.export.export import load_exported_arrays
    from training.models.resnet import build_resnet
    header, arrays = load_exported_arrays(os.path.join(models_dir(), f"resnet{n}-8plane-fp32.rune"))
    assert header["in_planes"] == 8
    assert header["board_size"] == n
    m = build_resnet(board=n, channels=8, blocks=blocks, in_planes=8, feature_set="go_planes_v02")
    state_dict = m.state_dict()
    name_map = {"stem_w": "stem.weight", "stem_b": "stem.bias", "vh1": "value.fc1.weight", "bh1": "value.fc1.bias", "wv": "value.fcv.weight", "bv": "value.fcv.bias", "wwdl": "value.fcwdl.weight", "bwdl": "value.fcwdl.bias", "wpol": "policy.fc.weight", "bpol": "policy.fc.bias"}
    for i in range(blocks):
        name_map[f"b{i}_w1"] = f"tower.{i}.c1.weight"
        name_map[f"b{i}_b1"] = f"tower.{i}.c1.bias"
        name_map[f"b{i}_w2"] = f"tower.{i}.c2.weight"
        name_map[f"b{i}_b2"] = f"tower.{i}.c2.bias"
    for k, v in arrays.items():
        if k in name_map:
            target = state_dict[name_map[k]]
            state_dict[name_map[k]] = torch.tensor(v).reshape(target.shape)
    m.load_state_dict(state_dict)
    m.eval()
    return m


def check_size(n, blocks, name):
    from training.games.go_v02 import GoGameV02
    with open(os.path.join(resnet_dir(), name)) as f:
        d = json.load(f)
    assert d["board"] == n
    assert d["in_planes"] == 8
    m = load_fixture_model(n, blocks)
    g = GoGameV02(size=n)
    for vec in d["vectors"]:
        planes = g.planes(vec["state"])
        assert list(planes.shape) == [1, 8, n, n]
        assert vec["legal_count"] == len(g.legal(vec["state"]))
        assert len(g.context(vec["state"])) == 12
        with torch.no_grad():
            v, _, _, probs = m(torch.tensor(planes))
        assert abs(float(v[0]) - vec["value"]) < 1e-5
        assert abs(float(probs.sum()) - 1.0) < 1e-5


def test_golden_13():
    check_size(13, 1, "eval_v02_13.json")


def test_golden_19():
    check_size(19, 1, "eval_v02_19.json")
