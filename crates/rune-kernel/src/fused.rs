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

use super::{mat_mul, mat_mul_tt, mat_vec, mat_vec_clipped};

pub fn qkv_fused_8x32(wq: &[f32], bq: &[f32], wk: &[f32], bk: &[f32], wv: &[f32], bv: &[f32], x: &[f32], q: &mut [f32], k: &mut [f32], v: &mut [f32]) {
    for t in 0..8 {
        let xb = &x[t * 32..t * 32 + 32];
        let qb = &mut q[t * 32..t * 32 + 32];
        let kb = &mut k[t * 32..t * 32 + 32];
        let vb = &mut v[t * 32..t * 32 + 32];
        mat_vec(wq, xb, Some(bq), qb, 32, 32);
        mat_vec(wk, xb, Some(bk), kb, 32, 32);
        mat_vec(wv, xb, Some(bv), vb, 32, 32);
    }
}

pub fn score_bias_gate_8x8(q: &[f32], k: &[f32], gab: &[f32], gate: crate::Gate, scores: &mut [f32], gate_out: &mut [f32]) {
    mat_mul_tt(q, k, scores, 8, 8, 32);
    for i in 0..64 {
        let b = scores[i] + gab[i];
        scores[i] = b;
        gate_out[i] = gate.apply(b);
    }
}

pub fn score_bias_gate_inplace(q: &[f32], k: &[f32], gab: &[f32], gate: crate::Gate, buf: &mut [f32]) {
    mat_mul_tt(q, k, buf, 8, 8, 32);
    for i in 0..64 {
        let b = buf[i] + gab[i];
        buf[i] = gate.apply(b);
    }
}

pub fn mix_residual_8x32(g: &[f32], v: &[f32], x: &[f32], alpha: f32, tmp: &mut [f32], out: &mut [f32]) {
    mat_mul(g, v, tmp, 8, 32, 8);
    super::specialized::residual_8x32(x, tmp, alpha, out);
}

pub fn linear_bias_clip_128x256(w1: &[f32], b1: &[f32], flat: &[f32], h1: &mut [f32]) {
    mat_vec_clipped(w1, flat, Some(b1), h1, 128, 256);
}

pub fn linear_bias_clip_32x128(w2: &[f32], b2: &[f32], h1: &[f32], h2: &mut [f32]) {
    mat_vec_clipped(w2, h1, Some(b2), h2, 32, 128);
}

pub fn dot_tanh(wvo: &[f32], bvo: f32, h2: &[f32]) -> f32 {
    let mut acc = bvo;
    for i in 0..h2.len() {
        acc += wvo[i] * h2[i];
    }
    acc.tanh()
}

pub fn wdl_3x32(wwdl: &[f32], bwdl: &[f32], h2: &[f32], wdl: &mut [f32; 3]) {
    for r in 0..3 {
        let mut acc = bwdl[r];
        for c in 0..32 {
            acc += wwdl[r * 32 + c] * h2[c];
        }
        wdl[r] = acc;
    }
}
