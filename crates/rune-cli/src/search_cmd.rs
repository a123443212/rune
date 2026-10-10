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

use std::cell::Cell;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;
use rune_runtime::board::Board;
use rune_runtime::evaluator::Evaluator;
use rune_search::lazy::{LazyConfig, LazyMode};

pub fn run(model: &str, fen: &str, depth: usize, lazy: &str, threshold: f32) -> i32 {
    let p = PathBuf::from(model);
    let mut ev = match Evaluator::load(&p) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("load failed: {}", e);
            return 1;
        }
    };
    let cfg = LazyConfig {
        mode: match lazy {
            "L1" => LazyMode::L1,
            "L2" => LazyMode::L2,
            _ => LazyMode::L0,
        },
        margin: 0.08,
        max_refine: 1,
        threshold,
    };
    let count = Rc::new(Cell::new(0usize));
    let cc = count.clone();
    let mut eval_fn = move |f: &str| {
        cc.set(cc.get() + 1);
        match Board::parse_fen(f) {
            Ok(b) => ev.evaluate_board(&b).value,
            Err(_) => 0.0,
        }
    };
    let t0 = Instant::now();
    let mut ab = rune_search::AlphaBeta::new(&mut eval_fn, cfg.clone());
    let (score, mv) = if cfg.mode == LazyMode::L0 {
        ab.search(fen, depth)
    } else {
        let mut ev_cheap = match Evaluator::load(&p) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("cheap load failed: {}", e);
                return 1;
            }
        };
        let mut cheap_fn = move |f: &str| {
            match Board::parse_fen(f) {
                Ok(b) => {
                    ev_cheap.refresh(&b);
                    ev_cheap.evaluate_value_only()
                }
                Err(_) => 0.0,
            }
        };
        ab.search_lazy(fen, depth, &mut cheap_fn)
    };
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("score {:.4}", score);
    println!("engine_score {}", rune_search::contract::engine_score(score));
    println!("move {}", mv.unwrap_or_default());
    println!("nodes {}", ab.stats.nodes);
    println!("evals {}", count.get());
    println!("cutoffs {}", ab.stats.cutoffs);
    println!("refined {}", ab.stats.refined);
    println!("qnodes {}", ab.stats.qnodes);
    println!("tt_hits {}", ab.stats.tt_hits);
    println!("ms {:.2}", ms);
    println!("nps {:.0}", ab.stats.nps());
    println!("root {} pv {} cut {} leaf {}", ab.stats.root_count, ab.stats.pv_count, ab.stats.cut_count, ab.stats.leaf_count);
    0
}
