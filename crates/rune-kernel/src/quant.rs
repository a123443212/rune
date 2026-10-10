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

use super::clipped_relu;

pub fn dequant_clip_i8(acc: &[i32], scale: f32, out: &mut [f32]) {
    debug_assert!(acc.len() == out.len());
    for i in 0..acc.len() {
        out[i] = clipped_relu(acc[i] as f32 * scale);
    }
}

pub fn dequant_clip_i8_batched(scales: &[f32; 8], dim: usize, acc: &[i32], out: &mut [f32]) {
    debug_assert!(acc.len() == 8 * dim);
    debug_assert!(out.len() == 8 * dim);
    for g in 0..8 {
        let s = scales[g];
        let base = g * dim;
        for d in 0..dim {
            out[base + d] = clipped_relu(acc[base + d] as f32 * s);
        }
    }
}

pub fn requant_row_int8(vals: &[f32], scale: f32, bound: i32, out: &mut [i32]) {
    for i in 0..vals.len() {
        let q = vals[i] / scale;
        let r = if q >= 0.0 { (q + 0.5).floor() } else { (q - 0.5).ceil() };
        let mut v = r as i64;
        if v > bound as i64 {
            v = bound as i64;
        }
        if v < -(bound as i64) {
            v = -(bound as i64);
        }
        out[i] = v as i32;
    }
}

pub fn clamp_row(out: &mut [f32], lo: f32, hi: f32) {
    for v in out.iter_mut() {
        if *v < lo {
            *v = lo;
        } else if *v > hi {
            *v = hi;
        }
    }
}
