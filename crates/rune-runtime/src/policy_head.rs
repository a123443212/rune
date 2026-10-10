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

use rune_kernel as kernel;

#[derive(Debug, Clone)]
pub struct PolicyWeights {
    pub input: usize,
    pub output: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

impl PolicyWeights {
    pub fn forward(&self, flat: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; self.output];
        kernel::mat_vec(&self.w, flat, Some(&self.b), &mut logits, self.output, self.input);
        let mut probs = vec![0.0f32; self.output];
        kernel::softmax(&logits, &mut probs);
        probs
    }

    pub fn logits(&self, flat: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.output];
        kernel::mat_vec(&self.w, flat, Some(&self.b), &mut out, self.output, self.input);
        out
    }
}
