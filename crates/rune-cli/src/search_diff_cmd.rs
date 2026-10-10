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
use rune_runtime::board::Board;
use rune_runtime::evaluator::Evaluator;
use rune_search::lazy::{LazyConfig, LazyMode};

pub fn run(model_a: &str, model_b: &str, fen: &str, depth: usize, tol: f32) -> i32 {
    let mut eva = match Evaluator::load(&PathBuf::from(model_a)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("load A failed: {}", e);
            return 1;
        }
    };
    let mut evb = match Evaluator::load(&PathBuf::from(model_b)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("load B failed: {}", e);
            return 1;
        }
    };
    let cfg = LazyConfig { mode: LazyMode::L0, margin: 0.08, max_refine: 1, threshold: 0.5 };
    let ca = Rc::new(Cell::new(0usize));
    let cb = Rc::new(Cell::new(0usize));
    let (sa, ma, na, apa) = {
        let cc = ca.clone();
        let mut eval_fn = move |fl: &str| {
            cc.set(cc.get() + 1);
            match Board::parse_fen(fl) {
                Ok(b) => eva.evaluate_board(&b).value,
                Err(_) => 0.0,
            }
        };
        let mut ab = rune_search::AlphaBeta::new(&mut eval_fn, cfg.clone());
        let (s, m) = ab.search(fen, depth);
        (s, m, ab.stats.nodes, ab.stats.cutoffs)
    };
    let (sb, mb, nb, cpb) = {
        let cc = cb.clone();
        let mut eval_fn = move |fl: &str| {
            cc.set(cc.get() + 1);
            match Board::parse_fen(fl) {
                Ok(b) => evb.evaluate_board(&b).value,
                Err(_) => 0.0,
            }
        };
        let mut ab = rune_search::AlphaBeta::new(&mut eval_fn, cfg);
        let (s, m) = ab.search(fen, depth);
        (s, m, ab.stats.nodes, ab.stats.cutoffs)
    };
    let d = (sa - sb).abs();
    let bad = d.is_nan();
    println!("a_score {:.4} a_move {} a_nodes {} a_evals {} a_cutoffs {}", sa, ma.clone().unwrap_or_default(), na, ca.get(), apa);
    println!("b_score {:.4} b_move {} b_nodes {} b_evals {} b_cutoffs {}", sb, mb.clone().unwrap_or_default(), nb, cb.get(), cpb);
    println!("node_delta {}", na.abs_diff(nb));
    println!("moves_match {}", ma == mb);
    if !bad && d <= tol {
        println!("status PARITY");
        0
    } else {
        println!("first_divergence ply 0 fen {} a {:.4} b {:.4} diff {:.2e}", fen, sa, sb, d);
        println!("status DIVERGED");
        2
    }
}
