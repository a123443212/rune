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

pub const MODEL_FORMAT_VERSION: u32 = 2;
pub const MAGIC: [u8; 4] = [82, 85, 78, 69];
pub const MAX_HEADER_LEN: usize = 1000000;
pub const MAX_TENSOR_ELEMS: usize = 200000000;
pub const MAX_PAYLOAD_BYTES: usize = 800000000;

pub fn is_supported_format(v: u32) -> bool {
    v == 1 || v == 2
}

pub fn is_supported_quant(q: &str) -> bool {
    q == "fp32" || q == "int8" || q == "int16"
}

pub fn quant_bound(quant: &str) -> i32 {
    if quant == "int16" {
        32767
    } else {
        127
    }
}
