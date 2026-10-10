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

pub fn normalize_policy_inplace(probs: &mut [f32]) {
    let mut s = 0.0f32;
    for v in probs.iter() {
        if *v > 0.0 && v.is_finite() {
            s += *v;
        }
    }
    if s <= 0.0 || !s.is_finite() {
        let u = 1.0 / probs.len().max(1) as f32;
        for v in probs.iter_mut() {
            *v = u;
        }
        return;
    }
    for v in probs.iter_mut() {
        if *v <= 0.0 || !v.is_finite() {
            *v = 0.0;
        } else {
            *v /= s;
        }
    }
}

pub fn mask_policy(policy: &[f32], legal: &[bool], out: &mut [f32]) {
    for i in 0..policy.len().min(legal.len()).min(out.len()) {
        out[i] = if legal[i] { policy[i].max(0.0) } else { 0.0 };
    }
    normalize_policy_inplace(out);
}

pub fn policy_entropy(probs: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for p in probs.iter() {
        if *p > 0.0 {
            h -= *p * p.ln();
        }
    }
    h
}
