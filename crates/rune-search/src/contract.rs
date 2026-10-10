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

#[derive(Debug, Clone, Default)]
pub struct EvalOutput {
    pub value: f32,
    pub wdl: [f32; 3],
    pub uncertainty: f32,
    pub refine: bool,
    pub policy: Vec<f32>,
    pub score_mean: f32,
}

pub fn canonical_value(raw: f32) -> f32 {
    if raw.is_nan() {
        return 0.0;
    }
    raw.clamp(-1.0, 1.0)
}

pub fn engine_score(value: f32) -> i32 {
    let v = canonical_value(value);
    let mut s = (v * 1000.0 + if v >= 0.0 { 0.5 } else { -0.5 }) as i32;
    if s >= 9000 {
        s = 8999;
    }
    if s <= -9000 {
        s = -8999;
    }
    s
}

pub fn normalize_wdl(w: &mut [f32; 3]) {
    let s = w[0] + w[1] + w[2];
    if s <= 0.0 {
        *w = [0.0, 1.0, 0.0];
        return;
    }
    w[0] /= s;
    w[1] /= s;
    w[2] /= s;
}

pub fn wdl_value(w: &[f32; 3]) -> f32 {
    let s = w[0] + w[1] + w[2];
    if s <= 0.0 {
        return 0.0;
    }
    (w[0] - w[2]) / s
}

pub fn value_wdl_consistent(value: f32, w: &[f32; 3], tol: f32) -> bool {
    (canonical_value(value) - wdl_value(w)).abs() <= tol
}

pub fn is_extreme(value: f32, bound: f32) -> bool {
    canonical_value(value).abs() >= bound
}
