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

use std::path::PathBuf;
use rune_runtime::board::Board;
use rune_runtime::compiled::CompiledEvaluator;
use rune_runtime::evaluator::Evaluator;

pub fn run(generic: &str, compiled: &str, positions: &str, tol: f32) -> i32 {
    let data = match std::fs::read_to_string(positions) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("positions failed: {}", e);
            return 1;
        }
    };
    let gp = PathBuf::from(generic);
    let cp = PathBuf::from(compiled);
    let mut ev_g = match Evaluator::load(&gp) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("generic load failed: {}", e);
            return 1;
        }
    };
    let mut ev_c = match CompiledEvaluator::load(&cp) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("compiled load failed: {}", e);
            return 1;
        }
    };
    let mut maxd: f32 = 0.0;
    let mut bad = false;
    let mut n = 0;
    for line in data.lines() {
        let fen = line.trim();
        if fen.is_empty() {
            continue;
        }
        let b = match Board::parse_fen(fen) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("bad fen {}: {}", fen, e);
                return 1;
            }
        };
        let r1 = ev_g.evaluate_board(&b);
        let r2 = ev_c.evaluate_board(&b);
        let d = (r1.value - r2.value).abs();
        if d.is_nan() {
            bad = true;
        } else if d > maxd {
            maxd = d;
        }
        println!("pos {} generic {:.6} compiled {:.6} diff {:.2e}", n, r1.value, r2.value, d);
        n += 1;
    }
    let pass = !bad && maxd <= tol;
    println!("compared {} max_abs_diff {:.9} tol {:.9} {}", n, maxd, tol, if pass { "PASS" } else { "FAIL" });
    if pass {
        0
    } else {
        2
    }
}
