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

use rune_spec as spec;

pub fn routing_refine(score: f32, threshold: f32, t_high: f32, has_t_low: bool, t_low: f32) -> bool {
    if score.is_nan() {
        return true;
    }
    if score >= t_high {
        return true;
    }
    if has_t_low && score < t_low {
        return false;
    }
    score >= threshold
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut m: f32 = 0.0;
    let mut ai: usize = 0;
    for i in 0..a.len().min(b.len()) {
        let d = (a[i] - b[i]).abs();
        if d > m {
            m = d;
            ai = i;
        }
    }
    (m, ai)
}

pub fn spec_check() -> bool {
    let _ = spec::FEATURE_VERSION;
    true
}
