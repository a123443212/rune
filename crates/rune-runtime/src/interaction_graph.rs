// RUNE — Relational Unified Neural Evaluator
// Copyright (C) 2026 a123443212
//
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// This project is dual-licensed under the MIT License and the
// Apache License, Version 2.0. You may choose either license
// when using, copying, modifying, or distributing this software.
//
// MIT License: https://opensource.org/license/mit
// Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, this
// software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
// OR CONDITIONS OF ANY KIND, either express or implied.

pub const GRAPH_VERSION: &str = "v13-graph-01";
pub const INVALIDATION_VERSION: &str = "v13-inv-01";

#[derive(Debug, Clone)]
pub struct InteractionGraph {
    pub tokens: usize,
    pub edges: Vec<(usize, usize)>,
}

impl InteractionGraph {
    pub fn dense(tokens: usize) -> InteractionGraph {
        let mut edges = Vec::with_capacity(tokens * tokens);
        for a in 0..tokens {
            for b in 0..tokens {
                edges.push((a, b));
            }
        }
        InteractionGraph { tokens, edges }
    }

    pub fn pruned(&self, keep: &[(usize, usize)]) -> InteractionGraph {
        InteractionGraph {
            tokens: self.tokens,
            edges: self.edges.iter().copied().filter(|e| keep.contains(e)).collect(),
        }
    }

    pub fn is_dense(&self) -> bool {
        self.edges.len() == self.tokens * self.tokens
    }

    pub fn affected_edges(&self, changed: &[usize]) -> Vec<(usize, usize)> {
        let mut mark = vec![false; self.tokens];
        for t in changed {
            if *t < self.tokens {
                mark[*t] = true;
            }
        }
        self.edges.iter().copied().filter(|(a, b)| mark[*a] || mark[*b]).collect()
    }

    pub fn score_cells_for_changed(tokens: usize, changed_len: usize) -> usize {
        let changed = changed_len.min(tokens);
        2 * changed * tokens - changed * changed
    }
}
