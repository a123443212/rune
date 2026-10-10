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

use rune_ir::{FusionGroup, RuneIr};

pub fn fuse_plan(ir: &mut RuneIr) {
    let mut fusion: Vec<FusionGroup> = Vec::new();
    let kinds: Vec<String> = ir.ops.iter().map(|o| o.kind.clone()).collect();
    let has = |k: &str| kinds.iter().any(|x| x == k);
    let fixed_8x32 = ir.model.tokens == 8 && ir.model.token_dim == 32;
    if fixed_8x32 && has("Q") && has("K") && has("V") {
        fusion.push(FusionGroup {
            id: "f_qkv".to_string(),
            ops: vec!["Q".to_string(), "K".to_string(), "V".to_string()],
            kernel: "qkv_fused_8x32".to_string(),
        });
    }
    if fixed_8x32 && has("Score") && has("Bias") && has("Gate") {
        fusion.push(FusionGroup {
            id: "f_score_bias_gate".to_string(),
            ops: vec!["Score".to_string(), "Bias".to_string(), "Gate".to_string()],
            kernel: "score_bias_gate_8x8".to_string(),
        });
    }
    if fixed_8x32 && has("Mix") && has("Residual") {
        fusion.push(FusionGroup {
            id: "f_mix_residual".to_string(),
            ops: vec!["Mix".to_string(), "Residual".to_string()],
            kernel: "mix_residual_8x32".to_string(),
        });
    }
    if has("HeadH1") {
        fusion.push(FusionGroup {
            id: "f_head_h1".to_string(),
            ops: vec!["HeadH1".to_string()],
            kernel: "linear_bias_clip".to_string(),
        });
    }
    if has("HeadH2") {
        fusion.push(FusionGroup {
            id: "f_head_h2".to_string(),
            ops: vec!["HeadH2".to_string()],
            kernel: "linear_bias_clip".to_string(),
        });
    }
    ir.fusion = fusion;
}
