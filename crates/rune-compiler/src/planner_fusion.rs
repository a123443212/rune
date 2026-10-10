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

pub fn fusion_of(kind: &str) -> String {
    match kind {
        "Q" | "K" | "V" => "f_qkv".to_string(),
        "Score" | "Bias" | "Gate" => "f_score_bias_gate".to_string(),
        "Mix" | "Residual" => "f_mix_residual".to_string(),
        "HeadH1" => "f_head_h1".to_string(),
        "HeadH2" => "f_head_h2".to_string(),
        "StemConv" | "Conv2D" => "f_conv".to_string(),
        "ResidualAdd" | "Relu" => "f_resblock".to_string(),
        "PolicyLogits" | "Policy" => "f_policy".to_string(),
        _ => String::new(),
    }
}
