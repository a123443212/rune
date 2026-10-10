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


def quantize_delta_int8(delta, scale):
    q = torch.round(delta / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def apply_quant_delta(state_int32, qdelta):
    return state_int32 + qdelta.to(torch.int32)


def dequant_tokens(acc_int32, scales):
    out = acc_int32.float() * scales.unsqueeze(-1)
    return torch.clamp(out, 0.0, 1.0)
