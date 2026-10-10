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

pub fn quantize_half_away(w: f32, scale: f32, bound: i32) -> i32 {
    if !(scale > 0.0) {
        return 0;
    }
    if w.is_nan() {
        return 0;
    }
    if w > 1e30 {
        return bound;
    }
    if w < -1e30 {
        return -bound;
    }
    let q = w / scale;
    let r = if q >= 0.0 { (q + 0.5).floor() } else { (q - 0.5).ceil() };
    let mut v = r as i64;
    if v > bound as i64 {
        v = bound as i64;
    }
    if v < -(bound as i64) {
        v = -(bound as i64);
    }
    v as i32
}

pub fn dequantize(q: i32, scale: f32) -> f32 {
    q as f32 * scale
}

pub fn symmetric_scale(table: &[f32], bound: i32) -> f32 {
    let mut m: f32 = 0.0;
    for v in table {
        let a = v.abs();
        if a > m {
            m = a;
        }
    }
    if m <= 0.0 {
        return 1.0;
    }
    m / bound as f32
}
