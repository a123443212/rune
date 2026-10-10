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

use super::version::{CLASSIC_REQUIRED_OPS, RESNET_REQUIRED_OPS};

pub fn arch_family(arch: &str) -> &'static str {
    if arch.starts_with("RUNE-RESNET") {
        "resnet"
    } else if arch == "RUNE-04" || arch == "RUNE-05" || arch == "RUNE-03" || arch.starts_with("RUNE-03-") {
        "legacy"
    } else if arch == "RUNE-SFNN"
        || arch == "RUNE-MLP"
        || arch == "RUNE-MLP-S"
        || arch == "RUNE-SFNN-C"
        || arch == "RUNE-ATTN"
        || arch == "RUNE-ATTN-GAB"
        || arch == "RUNE-ATTN-DUAL"
        || arch == "RUNE-ATTN-MH4"
        || arch == "RUNE-REL-02"
        || arch == "RUNE-REL-LITE"
    {
        "classic"
    } else {
        "unknown"
    }
}

pub fn required_ops_for_arch(arch: &str) -> Vec<&'static str> {
    match arch_family(arch) {
        "resnet" => RESNET_REQUIRED_OPS.to_vec(),
        "classic" => CLASSIC_REQUIRED_OPS.to_vec(),
        _ => Vec::new(),
    }
}

pub fn is_resnet_arch(arch: &str) -> bool {
    arch_family(arch) == "resnet"
}
