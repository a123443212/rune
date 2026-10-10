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

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class TrainingModel:
    model: nn.Module
    needs_context: bool
    is_adaptive: bool
    is_search: bool


_FROZEN_PREFIXES = ("RUNE-03-", "RUNE-04", "RUNE-05")
_FROZEN_IDS = ("RUNE-ATTN-DUAL", "RUNE-ATTN-MH4")


def build_training_model(config):
    architecture = config["arch"]
    if architecture == "RUNE-ATTN":
        architecture = "RUNE-ATTN-GAB"
    if architecture in _FROZEN_IDS or architecture.startswith(_FROZEN_PREFIXES):
        raise RuntimeError(
            f"{architecture} is frozen and removed from training; "
            "use RUNE-ATTN-GAB, RUNE-ATTN-SOFT, RUNE-MLP, RUNE-SFNN or RUNE-REL-02")
    head = config.get("head", "value_wdl")
    buckets = config.get("head_buckets", 1)
    game = config.get("game", "chess")
    pair = config.get("pair", False)

    if architecture == "RUNE-REL-LITE":
        from training.models.relational import build_rel_model

        params = config.get("rel_params", {})
        model = build_rel_model(
            tokens=6,
            dim=24,
            gate=params.get("gate", "clip"),
            alpha=params.get("alpha", 1.0),
            dynamic_bias=False,
            pair=params.get("pair", pair),
            head=head,
            game=game,
        )
        return TrainingModel(model, True, False, False)

    if architecture == "RUNE-MLP-S":
        from training.models.rune_models import build_model

        params = config.get("rel_params", {})
        model = build_model(
            "RUNE-MLP",
            gate=params.get("gate", "clip"),
            pair=pair,
            game=game,
            buckets=buckets,
            head=head,
            head_h1=64,
            head_h2=16,
        )
        return TrainingModel(model, False, False, False)

    if architecture == "RUNE-SFNN-C":
        from training.models.rune_models import build_model

        params = config.get("rel_params", {})
        model = build_model(
            "RUNE-SFNN",
            gate=params.get("gate", "clip"),
            pair=pair,
            game=game,
            buckets=buckets,
            head=head,
            head_h1=128,
            head_h2=16,
        )
        return TrainingModel(model, False, False, False)

    if architecture == "RUNE-REL-02":
        from training.models.relational import build_rel_model

        params = config.get("rel_params", {})
        model = build_rel_model(
            tokens=params.get("tokens", 8),
            dim=params.get("dim", 32),
            gate=params.get("gate", "clip"),
            alpha=params.get("alpha", 1.0),
            dynamic_bias=params.get("dynamic_bias", False),
            pair=params.get("pair", pair),
            head=head,
            game=game,
        )
        return TrainingModel(model, True, False, False)

    from training.models.rune_models import build_model

    params = config.get("rel_params", {})
    model = build_model(
        architecture,
        gate=params.get("gate", "clip"),
        pair=params.get("pair", pair),
        game=game,
        buckets=buckets,
        head=head,
        head_h1=config.get("head_h1"),
        head_h2=config.get("head_h2"),
    )
    return TrainingModel(model, False, False, False)
