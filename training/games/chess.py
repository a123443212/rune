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

from training.features import context as ctx_mod
from training.features import python_features as pf
from training.games.base import GameSpec


class ChessGame(GameSpec):
    game_id = "chess"
    num_groups = pf.NUM_GROUPS
    vocabs = tuple(pf.VOCAB_SIZES)
    tokens = pf.TOKENS
    token_dim = pf.TOKEN_DIM
    context_dim = ctx_mod.CONTEXT_DIM
    context_names = tuple(ctx_mod.CONTEXT_NAMES)
    feature_version = pf.FEATURE_VERSION

    def extract(self, state):
        return pf.extract_features(state)

    def context(self, state):
        return ctx_mod.context_vector(state)

    def phase(self, state):
        board, _, _, _ = pf.parse_fen(state)
        return pf.game_phase(board)

    def normalize(self, state):
        return pf.normalized_key(state)

    def phase_from_ids(self, group_ids, group_mask):
        g7 = group_ids[7].clamp(0, 63)
        valid = (group_mask[7] > 0.5) & (g7 >= 34) & (g7 <= 36)
        cand = torch.where(valid, g7, torch.full_like(g7, 34))
        return (cand.amax(dim=1) - 34).clamp(0, 2).long()
