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

use crate::mixer::MixerWeights;

#[derive(Debug, Clone)]
pub struct DualMixer {
    pub first: MixerWeights,
    pub second: MixerWeights,
}

impl DualMixer {
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        let n = x.len();
        let mut tmp = vec![0.0_f32; n];
        self.first.forward(x, None, &mut tmp, None);
        self.second.forward(&tmp, None, out, None);
    }
}
