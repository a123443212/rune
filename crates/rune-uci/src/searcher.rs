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

use std::time::Instant;

use rune_runtime::board::Board;
use rune_runtime::evaluator::Evaluator;
use rune_search::lazy::{LazyConfig, LazyMode};

pub struct SearchOutcome {
    pub bestmove: String,
    pub value: f32,
    pub nodes: usize,
    pub millis: u64,
}

pub fn value_to_cp(v: f32) -> i32 {
    let c = v.clamp(-0.9999, 0.9999);
    (-400.0 * ((1.0 - c) / (1.0 + c)).log10()).round() as i32
}

pub fn search_depth(ev: &mut Evaluator, fen: &str, depth: usize) -> SearchOutcome {
    let mut eval_fn = |f: &str| match Board::parse_fen(f) {
        Ok(b) => ev.evaluate_board(&b).value,
        Err(_) => 0.0,
    };
    let t0 = Instant::now();
    let mut ab = rune_search::AlphaBeta::new(&mut eval_fn, LazyConfig { mode: LazyMode::L0, margin: 0.08, max_refine: 1, threshold: 0.5 });
    let (score, mv) = ab.search(fen, depth);
    SearchOutcome { bestmove: mv.unwrap_or_default(), value: score, nodes: ab.stats.nodes, millis: t0.elapsed().as_millis() as u64 }
}

pub fn search_timed(ev: &mut Evaluator, fen: &str, budget_ms: u64, max_depth: usize) -> SearchOutcome {
    let t0 = Instant::now();
    let mut best = search_depth(ev, fen, 1);
    if t0.elapsed().as_millis() as u64 >= budget_ms {
        return best;
    }
    for d in 2..=max_depth.max(1) {
        if t0.elapsed().as_millis() as u64 >= budget_ms {
            break;
        }
        let r = search_depth(ev, fen, d);
        if r.bestmove.is_empty() {
            break;
        }
        best = r;
    }
    best
}
