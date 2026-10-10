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


def build_training_model(config):
    architecture = config["arch"]
    if architecture == "RUNE-REL-LITE":
        from training.models.rel_lite import build_rel_lite

        params = config.get("rel_params", {})
        model = build_rel_lite(
            gate=params.get("gate", "clip"),
            alpha=params.get("alpha", 1.0),
            pair=params.get("pair", config.get("pair", False)),
            game=config.get("game", "chess"),
        )
        return TrainingModel(model, True, False, False)

    if architecture == "RUNE-MLP-S":
        from training.models.mlp_small import build_mlp_small

        model = build_mlp_small(
            buckets=config.get("head_buckets", 1),
            pair=config.get("pair", False),
            game=config.get("game", "chess"),
        )
        return TrainingModel(model, False, False, False)

    if architecture == "RUNE-SFNN-C":
        from training.models.sfnn_compact import build_sfnn_compact

        model = build_sfnn_compact(
            buckets=config.get("head_buckets", 1),
            pair=config.get("pair", False),
            game=config.get("game", "chess"),
        )
        return TrainingModel(model, False, False, False)

    if architecture == "RUNE-ATTN-DUAL":
        from training.models.dual_attention import build_dual_attention

        params = config.get("rel_params", {})
        model = build_dual_attention(
            buckets=config.get("head_buckets", 1),
            gate=params.get("gate", "clip"),
            pair=params.get("pair", config.get("pair", False)),
            game=config.get("game", "chess"),
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
            pair=params.get("pair", config.get("pair", False)),
            game=config.get("game", "chess"),
        )
        return TrainingModel(model, True, False, False)

    if architecture.startswith("RUNE-03-"):
        from training.models.dense import build_dense_model

        params = config.get("dense_params", {})
        model = build_dense_model(
            variant=architecture.split("-")[-1],
            token_dims=params.get("token_dims", [32] * 8),
            pooling=params.get("pooling", "none"),
            pool_clip=params.get("pool_clip", True),
            gate_on=params.get("gate_on", False),
            shared_width=params.get("shared_width", 32),
            head_h1=params.get("head_h1", 128),
            head_h2=params.get("head_h2", 32),
        )
        return TrainingModel(model, False, False, False)

    if architecture.startswith("RUNE-04"):
        from training.models.adaptive import build_adaptive_model

        params = config.get("adaptive_params", {})
        model = build_adaptive_model(
            dim=params.get("dim", 32),
            cheap_pooling=params.get("cheap_pooling", "none"),
            alpha=params.get("alpha", 1.0),
            threshold=params.get("threshold", 0.5),
            t_high=params.get("t_high"),
            t_low=params.get("t_low"),
            pruned_pairs=params.get("pruned_pairs", ()),
            refine_precision=params.get("refine_precision", "fp32"),
            cheap_hidden=params.get("cheap_hidden", 32),
            ref_h1=params.get("ref_h1", 128),
            ref_h2=params.get("ref_h2", 32),
        )
        return TrainingModel(model, False, True, False)

    if architecture.startswith("RUNE-05"):
        from training.models.uncertainty import build_search_model

        params = config.get("adaptive_params", {})
        model = build_search_model(
            dim=params.get("dim", 32),
            cheap_pooling=params.get("cheap_pooling", "none"),
            alpha=params.get("alpha", 1.0),
            threshold=params.get("threshold", 0.5),
            t_high=params.get("t_high"),
            t_low=params.get("t_low"),
            pruned_pairs=params.get("pruned_pairs", ()),
            refine_precision=params.get("refine_precision", "fp32"),
            uncertainty_on=True,
            stability_on=config.get("lambda_stab", 0.0) > 0,
            cheap_hidden=params.get("cheap_hidden", 32),
            ref_h1=params.get("ref_h1", 128),
            ref_h2=params.get("ref_h2", 32),
        )
        return TrainingModel(model, False, True, True)

    from training.models.rune_models import build_model

    params = config.get("rel_params", {})
    model = build_model(
        architecture,
        gate=params.get("gate", "clip"),
        pair=params.get("pair", config.get("pair", False)),
        game=config.get("game", "chess"),
        buckets=config.get("head_buckets", 1),
    )
    return TrainingModel(model, False, False, False)