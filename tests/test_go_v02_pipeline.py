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

import torch


def test_adapter_callbacks():
    from training.games.go_v02 import GoGameV02
    from training.models.resnet import build_resnet
    from training.search.go_adapter import build_go_callbacks, move_to_index, priors_for_legal
    torch.manual_seed(3)
    g = GoGameV02(size=9)
    m = build_resnet(board=9, channels=8, blocks=1, in_planes=8, feature_set="go_planes_v02")
    m.eval()
    s = "/".join(["." * 9] * 9) + " b - 7.5 1 0"
    legal_fn, apply_fn, eval_fn = build_go_callbacks(g, m)
    legal = legal_fn(s)
    assert len(legal) == 82
    assert -1 in legal
    assert move_to_index(-1, 9) == 81
    v, p = eval_fn(s)
    assert -1.0 <= v <= 1.0
    assert len(p) == len(legal)
    assert abs(sum(p) - 1.0) < 1e-5
    assert abs(sum(priors_for_legal([0.0] * 82, legal, 9)) - 1.0) < 1e-9
    nxt = apply_fn(s, legal[0])
    assert nxt != s


def test_adapter_search_move():
    from training.games.go_v02 import GoGameV02
    from training.models.resnet import build_resnet
    from training.search.go_adapter import search_move
    torch.manual_seed(3)
    g = GoGameV02(size=9)
    m = build_resnet(board=9, channels=8, blocks=1, in_planes=8, feature_set="go_planes_v02")
    m.eval()
    s = "/".join(["." * 9] * 9) + " b - 7.5 1 0"
    mv, info = search_move(g, m, s, simulations=4, seed=1)
    assert mv in g.legal(s)
    assert info["root_visits"] > 0


def test_teacher_labels_and_costs():
    from training.engine.go_teacher import label_states, teacher_cost
    from training.games.go_v02 import GoGameV02
    g = GoGameV02(size=9)
    states = ["/".join(["." * 9] * 9) + " b - 7.5 1 0"]
    recs = label_states(g, states)
    assert len(recs) == 1
    r = recs[0]
    assert r["game"] == "go"
    assert r["teacher"] == "synthetic_go_v02"
    assert -1.0 <= r["value"] <= 1.0
    assert abs(sum(r["wdl"]) - 1.0) < 1e-9
    assert abs(sum(r["policy"]) - 1.0) < 1e-9
    c = teacher_cost(100)
    assert c["positions"] == 100
    assert c["labeling_seconds"] > 0.0
    assert c["storage_bytes"] > 0


def test_go_dataset_loader():
    from training.datasets.go_dataset import make_go_loader
    from training.engine.go_teacher import label_states
    from training.games.go_v02 import GoGameV02
    g = GoGameV02(size=9)
    states = ["/".join(["." * 9] * 9) + " b - 7.5 1 0", "/".join(["." * 9] * 9) + " w - 7.5 5 0"]
    recs = label_states(g, states)
    recs[0]["move"] = -1
    loader, ds = make_go_loader(recs, g, batch_size=2, shuffle=False, seed=0)
    assert len(ds) == 2
    batch = next(iter(loader))
    planes, value, wdl, policy = batch
    assert planes.shape == (2, 8, 9, 9)
    assert value.shape == (2,)
    assert wdl.shape == (2, 3)
    assert policy.shape == (2, 82)
    assert abs(float(policy[0].sum()) - 1.0) < 1e-5
