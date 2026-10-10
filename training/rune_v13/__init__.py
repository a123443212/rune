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

from training.rune_v13.token_delta import detect_changed_groups, group_deltas_to_tokens
from training.rune_v13.interaction_graph import InteractionGraph, build_dense_graph
from training.rune_v13.relational_reference import dense_forward, split_qkv_score_gate_mix
from training.rune_v13.incremental_state import IncrementalRelationalState, PathSelector
from training.rune_v13.quant_delta import quantize_delta_int8, apply_quant_delta

__all__ = [
    "detect_changed_groups",
    "group_deltas_to_tokens",
    "InteractionGraph",
    "build_dense_graph",
    "dense_forward",
    "split_qkv_score_gate_mix",
    "IncrementalRelationalState",
    "PathSelector",
    "quantize_delta_int8",
    "apply_quant_delta",
]
