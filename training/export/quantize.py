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

import numpy as np


def fake_quantize(arr, scale=None, bits=8):
    bound = 32767 if bits == 16 else 127
    arr = np.asarray(arr, dtype=np.float64)
    if scale is None:
        m = float(np.abs(arr).max()) if arr.size else 0.0
        scale = m / bound if m > 0 else 1.0
    flat = arr / scale
    r = np.where(flat >= 0, np.floor(flat + 0.5), np.ceil(flat - 0.5))
    q = np.clip(r, -bound, bound)
    return (q * scale).astype(np.float32), scale


def quantization_report(arrays):
    report = {}
    for name, arr in arrays.items():
        fq, scale = fake_quantize(np.asarray(arr, dtype=np.float32))
        err = np.abs(np.asarray(arr, dtype=np.float32) - fq)
        report[name] = {
            "scale": float(scale),
            "max_abs_err": float(err.max()) if err.size else 0.0,
            "mean_abs_err": float(err.mean()) if err.size else 0.0,
        }
    return report
