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

pub fn conv2d_nchw(input: &[f32], weight: &[f32], bias: Option<&[f32]>, out: &mut [f32], n: usize, cin: usize, cout: usize, h: usize, w: usize, kh: usize, kw: usize, pad_h: usize, pad_w: usize) {
    let oh = h + 2 * pad_h - kh + 1;
    let ow = w + 2 * pad_w - kw + 1;
    for nn in 0..n {
        for co in 0..cout {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc: f32 = if let Some(b) = bias { b[co] } else { 0.0 };
                    for ci in 0..cin {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = oy as isize + ky as isize - pad_h as isize;
                                let ix = ox as isize + kx as isize - pad_w as isize;
                                if iy < 0 || ix < 0 || iy >= h as isize || ix >= w as isize {
                                    continue;
                                }
                                let iv = input[((nn * cin + ci) * h + iy as usize) * w + ix as usize];
                                let wv = weight[((co * cin + ci) * kh + ky) * kw + kx];
                                acc += iv * wv;
                            }
                        }
                    }
                    out[((nn * cout + co) * oh + oy) * ow + ox] = acc;
                }
            }
        }
    }
}

pub fn residual_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert!(a.len() == b.len());
    debug_assert!(a.len() == out.len());
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

pub fn residual_add_relu_inplace(base: &[f32], delta: &mut [f32]) {
    debug_assert!(base.len() == delta.len());
    for i in 0..base.len() {
        let s = base[i] + delta[i];
        delta[i] = if s < 0.0 { 0.0 } else { s };
    }
}

pub fn residual_add_relu(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert!(a.len() == b.len());
    debug_assert!(a.len() == out.len());
    for i in 0..a.len() {
        let s = a[i] + b[i];
        out[i] = if s < 0.0 { 0.0 } else { s };
    }
}

pub fn global_avg_pool(input: &[f32], out: &mut [f32], n: usize, c: usize, h: usize, w: usize) {
    let hw = (h * w) as f32;
    for nn in 0..n {
        for cc in 0..c {
            let mut acc = 0.0f32;
            for y in 0..h {
                for x in 0..w {
                    acc += input[((nn * c + cc) * h + y) * w + x];
                }
            }
            out[nn * c + cc] = acc / hw;
        }
    }
}
