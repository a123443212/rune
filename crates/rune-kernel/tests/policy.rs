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

use rune_kernel::{normalize_policy, policy_entropy, softmax};

#[test]
fn softmax_sums_to_one() {
    let logits = vec![1.0f32, 2.0, 3.0];
    let mut out = vec![0.0f32; 3];
    softmax(&logits, &mut out);
    let s: f32 = out.iter().sum();
    assert!((s - 1.0).abs() < 1e-6);
    assert!(out[2] > out[1] && out[1] > out[0]);
}

#[test]
fn masked_policy_zeros_illegal() {
    let logits = vec![10.0f32, 0.0, 0.0];
    let legal = vec![false, true, true];
    let mut out = vec![0.0f32; 3];
    normalize_policy(&logits, &legal, &mut out);
    assert_eq!(out[0], 0.0);
    assert!((out[1] + out[2] - 1.0).abs() < 1e-6);
}

#[test]
fn entropy_uniform_max() {
    let u = vec![0.25f32; 4];
    let peaked = vec![1.0f32, 0.0, 0.0, 0.0];
    assert!(policy_entropy(&u) > policy_entropy(&peaked));
}
