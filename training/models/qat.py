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


def fake_quantize(x, bits=8):
    bound = 32767 if bits == 16 else 127
    m = float(x.detach().abs().max().item()) if x.numel() else 0.0
    if m <= 0.0:
        return x
    scale = m / float(bound)
    q = torch.clamp(torch.round(x / scale), -bound, bound)
    return q * scale


def fake_quantize_model(model, bits=8, skip=()):
    for name, p in model.named_parameters():
        if any(k in name for k in skip):
            continue
        if "emb" in name or "tables" in name:
            continue
        with torch.no_grad():
            p.copy_(fake_quantize(p, bits))


def sym_scales(state, bits=8):
    bound = 32767 if bits == 16 else 127
    out = {}
    for k, v in state.items():
        t = v.detach().float()
        m = float(t.abs().max().item()) if t.numel() else 0.0
        out[k] = m / float(bound) if m > 0.0 else 1.0
    return out
