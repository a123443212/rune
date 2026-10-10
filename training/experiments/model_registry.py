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

import warnings


MODEL_IDS = {
    "rune_mlp": "RUNE-MLP",
    "rune_mlp_swiglu": "RUNE-MLP",
    "rune_attn_gab": "RUNE-ATTN-GAB",
    "rune_attn_gab_pair": "RUNE-ATTN-GAB",
    "rune_attn_gab_swiglu": "RUNE-ATTN-GAB",
    "rune_attn_soft": "RUNE-ATTN-SOFT",
    "rune_attn_soft_pair": "RUNE-ATTN-SOFT",
    "rune_sfnn": "RUNE-SFNN",
    "rune_rel_s": "RUNE-REL-02",
    "rune_rel_d": "RUNE-REL-02",
    "rune_rel_swiglu": "RUNE-REL-02",
}

CORE_MODELS = ("rune_mlp",)
EXPERIMENTAL_MODELS = (
    "rune_attn_gab", "rune_rel_s", "rune_rel_d",
    "rune_attn_gab_pair", "rune_mlp_swiglu", "rune_attn_gab_swiglu",
    "rune_attn_soft", "rune_attn_soft_pair", "rune_sfnn", "rune_rel_swiglu",
)
PAIR_MODELS = frozenset(("rune_attn_gab_pair", "rune_attn_soft_pair"))

SWIGLU_MODELS = frozenset(("rune_mlp_swiglu", "rune_attn_gab_swiglu", "rune_rel_swiglu"))

ADAPTIVE_MODES = {}

SEARCH_ROUTING = {}

CORE_ARCH = {"tokens": 8, "token_dim": 32, "gate": "clip", "alpha": 1.0}


def pair_enabled_for_model(config, model_key):
    for leg in config.get("pair_legs", {}).values():
        if leg.get("model") == model_key:
            return bool(leg.get("pair", False))
    if model_key in PAIR_MODELS:
        return True
    models = config.get("models", [])
    return len(models) == 1 and bool(config.get("architecture", {}).get("pair", False))


def experimental_reasons(config):
    reasons = []
    for model in config.get("models", []):
        if model in EXPERIMENTAL_MODELS:
            reasons.append(f"model {model} is experimental")
    loss = config.get("loss", {})
    if loss.get("ranking", False):
        reasons.append("ranking loss is experimental (use the L1 driver on candidates)")
    if loss.get("uncertainty", False):
        reasons.append("uncertainty loss is experimental (L1 leg only, calibration-gated)")
    if loss.get("stability", False):
        reasons.append("stability loss is experimental (L2 leg only, needs child data)")
    dist = config.get("distillation", {})
    if dist.get("enabled", False):
        reasons.append(f"distillation is experimental (task={dist.get('task', 'value_wdl')}, "
                       f"weighting={dist.get('weight_mode', 'uniform')})")
    sampling = config.get("sampling", {})
    if sampling.get("mode", "none") == "disagreement_mix":
        reasons.append("disagreement sampling is experimental (stage-gated after S0)")
    architecture = config.get("architecture", {})
    for key in ("tokens", "token_dim", "gate", "alpha"):
        if architecture.get(key, CORE_ARCH[key]) != CORE_ARCH[key]:
            warnings.warn(f"architecture.{key}={architecture.get(key)} deviates from core")
    return reasons