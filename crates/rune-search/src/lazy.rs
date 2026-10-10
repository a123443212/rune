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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LazyMode {
    L0,
    L1,
    L2,
}

#[derive(Debug, Clone)]
pub struct LazyConfig {
    pub mode: LazyMode,
    pub margin: f32,
    pub max_refine: usize,
    pub threshold: f32,
}

impl Default for LazyConfig {
    fn default() -> Self {
        LazyConfig { mode: LazyMode::L0, margin: 0.08, max_refine: 1, threshold: 0.5 }
    }
}

pub fn should_refine(cheap: f32, alpha: f32, beta: f32, uncertainty: f32, threshold: f32, cfg: &LazyConfig) -> bool {
    match cfg.mode {
        LazyMode::L0 => true,
        LazyMode::L1 => {
            if cheap.is_nan() {
                return true;
            }
            cheap >= threshold
        }
        LazyMode::L2 => {
            if cheap.is_nan() {
                return true;
            }
            if cheap <= alpha - cfg.margin || cheap >= beta + cfg.margin {
                return false;
            }
            if uncertainty >= 0.5 {
                return true;
            }
            cheap >= threshold
        }
    }
}
