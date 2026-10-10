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


def build_precomputed(model_spec, arrays, scales):
    consts = {}
    quant = model_spec.get("quantization", "fp32")
    consts["quantization"] = quant
    consts["quant_scales"] = dict(scales)
    consts["zero_points"] = {k: 0 for k in scales.keys()}
    if quant in ("int8", "int16"):
        bound = 32767 if quant == "int16" else 127
        consts["quant_bound"] = bound
        consts["accum_width"] = "int32"
        consts["clamp"] = [0.0, 1.0]
    else:
        consts["quant_bound"] = None
        consts["accum_width"] = "fp32"
        consts["clamp"] = [0.0, 1.0]
    for key in ("gabS", "gab"):
        if key in arrays:
            g = np.asarray(arrays[key], dtype=np.float32)
            consts["static_gab_shape"] = list(g.shape)
            consts["static_gab_max"] = float(np.abs(g).max()) if g.size else 0.0
            break
    consts["routing_threshold"] = float(model_spec.get("threshold", 0.5))
    consts["routing_t_high"] = float(model_spec.get("t_high", model_spec.get("threshold", 0.5)))
    tlow = model_spec.get("t_low", None)
    consts["routing_t_low"] = None if tlow is None else float(tlow)
    consts["alpha"] = float(model_spec.get("alpha", 1.0))
    consts["bias_clamp"] = [-0.25, 0.25]
    consts["head_clamp"] = [0.0, 1.0]
    return consts
