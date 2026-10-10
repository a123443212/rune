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

class GameSpec:
    game_id = ""
    num_groups = 0
    vocabs = ()
    tokens = 8
    token_dim = 32
    context_dim = 0
    context_names = ()
    feature_version = ""

    def extract(self, state):
        raise NotImplementedError

    def context(self, state):
        raise NotImplementedError

    def phase(self, state):
        raise NotImplementedError

    def normalize(self, state):
        return state

    def phase_from_ids(self, group_ids, group_mask):
        raise NotImplementedError
