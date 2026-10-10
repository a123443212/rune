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

pub const IR_VERSION: &str = "1.1";
pub const SPEC_VERSION: &str = "RUNE-12";
pub const COMPILER_VERSION: &str = "0.12.0";
pub const PACKING_VERSION: u32 = 1;
pub const ACCEPTED_IR_VERSIONS: [&str; 2] = ["1.0", "1.1"];
pub const ACCEPTED_SPEC_VERSIONS: [&str; 3] = ["RUNE-10", "RUNE-11", "RUNE-12"];
pub const CLASSIC_REQUIRED_OPS: [&str; 15] = ["FeatureUpdate", "AccumulatorUpdate", "Tokenize", "Q", "K", "V", "Score", "Bias", "Gate", "Mix", "Residual", "HeadH1", "HeadH2", "Value", "WDL"];
pub const RESNET_REQUIRED_OPS: [&str; 8] = ["StemConv", "Conv2D", "ResidualAdd", "Relu", "Value", "WDL", "PolicyLogits", "Policy"];
pub const RESNET_OPTIONAL_OPS: [&str; 3] = ["GlobalPool", "Flatten", "FeaturePlanes"];
pub const RESNET_OP_KINDS: [&str; 11] = ["StemConv", "Conv2D", "ResidualAdd", "Relu", "GlobalPool", "Flatten", "FeaturePlanes", "PolicyLogits", "Policy", "Value", "WDL"];
