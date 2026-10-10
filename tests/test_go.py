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

def test_go_registered():
    from training.games import available, get
    assert "go" in available()
    g = get("go")
    assert g.feature_version == "go_planes_v01"


def test_go_extract_planes():
    from training.games import get
    g = get("go")
    rows = ["." * 9 for _ in range(9)]
    rows[4] = "....X...."
    state = "/".join(rows) + " b"
    feats = g.extract(state)
    assert len(feats) > 0
    planes = g.planes(state)
    assert planes.shape == (1, 1, 9, 9)
    assert planes[0, 0, 4, 4] == 1.0
    ctx = g.context(state)
    assert len(ctx) == 12


def test_go_empty():
    from training.games import get
    g = get("go")
    state = "/".join(["." * 9] * 9) + " b"
    planes = g.planes(state)
    assert float(abs(planes).sum()) == 0.0
