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

GRAPH_VERSION = "v13-graph-01"
INVALIDATION_VERSION = "v13-inv-01"


class InteractionGraph:
    def __init__(self, tokens, edges=None):
        self.tokens = int(tokens)
        self.version = GRAPH_VERSION
        if edges is None:
            edges = [(a, b) for a in range(self.tokens) for b in range(self.tokens)]
        self.edges = [(int(a), int(b)) for a, b in edges]
        self._index = {e: k for k, e in enumerate(self.edges)}

    def num_edges(self):
        return len(self.edges)

    def is_dense(self):
        return len(self.edges) == self.tokens * self.tokens

    def affected_edges(self, changed_tokens):
        changed = set(int(t) for t in changed_tokens)
        return [e for e in self.edges if e[0] in changed or e[1] in changed]

    def affected_rows(self, changed_tokens):
        return sorted(set(int(t) for t in changed_tokens))

    def edge_meta(self, a, b):
        return {
            "source": int(a),
            "target": int(b),
            "projection": "qkv_dot",
            "bias": "gab_plus_dynamic",
            "gate": "clip_or_hard_sigmoid",
            "value": "gate_times_v",
        }

    def pruned(self, keep):
        keep = set((int(a), int(b)) for a, b in keep)
        return InteractionGraph(self.tokens, [e for e in self.edges if e in keep])


def build_dense_graph(tokens=8):
    return InteractionGraph(tokens)
