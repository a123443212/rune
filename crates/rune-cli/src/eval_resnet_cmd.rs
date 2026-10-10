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
use rune_runtime::evaluator::Evaluator;

pub fn run_eval_resnet(model: &str, state: &str, size: usize) -> i32 {
    let p = PathBuf::from(model);
    let ev = match Evaluator::load(&p) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("load failed: {}", e);
            return 1;
        }
    };
    if !ev.is_resnet() {
        eprintln!("not a resnet model: {}", ev.arch_id());
        return 1;
    }
    let go = match rune_runtime::go::GoBoard::parse(state) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("bad go state: {}", e);
            return 1;
        }
    };
    if go.size != size && size != 0 {
        eprintln!("size mismatch {} vs {}", go.size, size);
        return 1;
    }
    let planes = rune_runtime::go::extract_planes(&go);
    let r = ev.evaluate_planes(&planes);
    println!("value {:.6}", r.value);
    println!("wdl {:.6} {:.6} {:.6}", r.wdl[0], r.wdl[1], r.wdl[2]);
    println!("policy_len {}", r.policy.len());
    let mut top: Vec<(usize, f32)> = r.policy.iter().cloned().enumerate().collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (i, v) in top.iter().take(5) {
        println!("p {} {:.6}", i, v);
    }
    println!("arch {}", ev.arch_id());
    0
}
