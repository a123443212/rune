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

use crate::relational_cache::{IncrWeights, RelationalCache};

pub fn dense_baseline(weights: &IncrWeights, tokens: &[f32]) -> Vec<f32> {
    let mut c = RelationalCache::configure(weights.clone(), usize::MAX);
    c.rebuild(tokens, None);
    c.out().to_vec()
}

pub fn b1_token_only_cost(_tokens: usize, dim: usize, changed: usize) -> usize {
    changed * dim * dim * 3
}

pub fn b2_cells_saved(tokens: usize, changed: usize) -> usize {
    tokens * tokens - (2 * changed * tokens - changed * changed)
}
