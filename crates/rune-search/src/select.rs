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

use crate::mcts::Node;

pub(crate) fn select_child(nodes: &[Node], idx: usize, cpuct: f32, fpu: f32) -> usize {
    let node = &nodes[idx];
    let mut total_visits = 0u32;
    for c in node.children.iter() {
        total_visits += nodes[*c].visits;
    }
    let total = (total_visits as f32).sqrt();
    let base = node.value + fpu;
    let mut best = node.children[0];
    let mut best_score = f32::NEG_INFINITY;
    for c in node.children.iter() {
        let child = &nodes[*c];
        let q = if child.visits == 0 { base } else { child.total / child.visits as f32 };
        let s = q + cpuct * child.prior * total / (1.0 + child.visits as f32);
        if s > best_score {
            best_score = s;
            best = *c;
        }
    }
    best
}
