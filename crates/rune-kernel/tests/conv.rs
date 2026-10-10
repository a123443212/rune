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

use rune_kernel::{conv2d_nchw, global_avg_pool, residual_add, residual_add_relu};

#[test]
fn conv_identity_1x1() {
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let w = vec![1.0f32];
    let out_size = 4;
    let mut out = vec![0.0f32; out_size];
    conv2d_nchw(&x, &w, None, &mut out, 1, 1, 1, 2, 2, 1, 1, 0, 0);
    assert_eq!(out, x);
}

#[test]
fn conv_3x3_center() {
    let x = vec![1.0f32; 9];
    let w = vec![1.0f32 / 9.0; 9];
    let mut out = vec![0.0f32; 1];
    conv2d_nchw(&x, &w, None, &mut out, 1, 1, 1, 3, 3, 3, 3, 0, 0);
    assert!((out[0] - 1.0).abs() < 1e-5);
}

#[test]
fn residual_add_relu_clamps() {
    let a = vec![1.0f32, -2.0];
    let b = vec![0.5f32, 1.0];
    let mut out = vec![0.0f32; 2];
    residual_add_relu(&a, &b, &mut out);
    assert_eq!(out, vec![1.5, 0.0]);
    residual_add(&a, &b, &mut out);
    assert_eq!(out, vec![1.5, -1.0]);
}

#[test]
fn avg_pool_uniform() {
    let x = vec![2.0f32; 16];
    let mut out = vec![0.0f32; 1];
    global_avg_pool(&x, &mut out, 1, 1, 4, 4);
    assert!((out[0] - 2.0).abs() < 1e-6);
}
