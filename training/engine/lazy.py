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

class LazyConfig:
    def __init__(self, mode="L0", margin=0.08, max_refine=1, fallback="full"):
        self.mode = str(mode)
        self.margin = float(margin)
        self.max_refine = int(max_refine)
        self.fallback = str(fallback)

    def to_dict(self):
        return {"mode": self.mode, "margin": self.margin, "max_refine": self.max_refine, "fallback": self.fallback}


def should_refine(cheap_value, alpha, beta, uncertainty, threshold, cfg):
    if cfg.mode == "L0":
        return True
    if cheap_value != cheap_value:
        return True
    if cfg.mode == "L1":
        return cheap_value >= threshold
    lo = alpha - cfg.margin
    hi = beta + cfg.margin
    if cheap_value <= lo or cheap_value >= hi:
        return False
    if uncertainty >= 0.5:
        return True
    return cheap_value >= threshold


def bounded_refine(n_refined, cfg):
    return n_refined < cfg.max_refine
