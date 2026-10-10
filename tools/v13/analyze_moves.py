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

import sys

import torch

sys.path.insert(0, ".")

from tests.test_v13_incremental import MOVE_PAIRS
from training.features.python_features import extract_features
from training.rune_v13 import build_dense_graph, detect_changed_groups, group_deltas_to_tokens
from training.features.python_features import NUM_GROUPS


def main():
    torch.manual_seed(3)
    tables = [torch.randn(512, 32) * 0.01 for _ in range(NUM_GROUPS)]
    g = build_dense_graph(8)
    print("move,changed_groups,changed_tokens,affected_cells,total_cells")
    for name, (fa, fb) in MOVE_PAIRS.items():
        a = extract_features(fa)
        b = extract_features(fb)
        changed, _, _ = detect_changed_groups(a, b)
        r = group_deltas_to_tokens(tables, a, b, 32)
        ct = r["changed_tokens"]
        cells = len(g.affected_edges(ct))
        print(f"{name},{len(changed)},{len(ct)},{cells},64")


if __name__ == "__main__":
    main()
