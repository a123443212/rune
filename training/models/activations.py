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


def clip01(x):
    return torch.clamp(x, 0.0, 1.0)


def apply_gate(name, scores):
    if name == "hard_sigmoid":
        return torch.clamp(0.2 * scores + 0.5, 0.0, 1.0)
    if name == "screlu":
        clipped = torch.clamp(scores, 0.0, 1.0)
        return clipped * clipped
    return clip01(scores)