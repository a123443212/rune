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

use std::collections::BTreeMap;
use rune_ir::{BufferPlace, MemoryPlan};

fn align_up(n: usize, a: usize) -> usize {
    ((n + a - 1) / a) * a
}

fn place_live(live: Vec<(&str, usize)>, groups: Vec<Vec<&str>>, strategy: String, in_place: Vec<String>) -> MemoryPlan {
    let mut places: BTreeMap<String, BufferPlace> = BTreeMap::new();
    let mut off: usize = 0;
    for (name, elems) in live {
        let mut shared_off: Option<usize> = None;
        for g in &groups {
            if g.contains(&name) {
                for other in g {
                    if *other != name {
                        if let Some(p) = places.get(*other) {
                            if p.elems >= elems {
                                shared_off = Some(p.offset);
                                break;
                            }
                        }
                    }
                }
            }
            if shared_off.is_some() {
                break;
            }
        }
        if let Some(s) = shared_off {
            places.insert(name.to_string(), BufferPlace { offset: s, elems, bytes: elems * 4, shared: true });
        } else {
            let s = align_up(off, 32);
            places.insert(name.to_string(), BufferPlace { offset: s, elems, bytes: elems * 4, shared: false });
            off = s + elems * 4;
        }
    }
    MemoryPlan { arena_bytes: align_up(off, 32), alignment: 32, buffers: places, strategy, in_place }
}

pub fn plan_memory(tokens: usize, dim: usize, h1: usize, h2: usize) -> MemoryPlan {
    let live: Vec<(&str, usize)> = vec![("tokens_buf", tokens * dim), ("q_buf", tokens * dim), ("k_buf", tokens * dim), ("v_buf", tokens * dim), ("scores_buf", tokens * tokens), ("gate_buf", tokens * tokens), ("mixed_raw_buf", tokens * dim), ("mixed_buf", tokens * dim), ("h1_buf", h1), ("h2_buf", h2), ("flat_buf", tokens * dim), ("tmp_row", dim.max(h1).max(h2))];
    let groups: Vec<Vec<&str>> = vec![vec!["q_buf", "h1_buf"], vec!["k_buf", "h2_buf"], vec!["scores_buf", "gate_buf"]];
    place_live(live, groups, "reuse q/h1, k/h2, scores/gate; single bump arena, no per-eval alloc".to_string(), vec!["gate_in_scores".to_string(), "mixed_raw_into_mixed_when_alpha_1".to_string()])
}

pub fn plan_memory_resnet(board: usize, channels: usize, policy: usize, h2: usize) -> MemoryPlan {
    let hw = board * board;
    let live: Vec<(&str, usize)> = vec![("planes_buf", hw), ("conv_buf_a", channels * hw), ("conv_buf_b", channels * hw), ("pooled_buf", channels), ("flat_buf", channels * hw), ("h_buf", h2), ("logits_buf", policy)];
    let groups: Vec<Vec<&str>> = vec![vec!["conv_buf_a", "conv_buf_b"]];
    place_live(live, groups, "resnet double-buffer conv; single bump arena".to_string(), vec!["relu_inplace".to_string()])
}
