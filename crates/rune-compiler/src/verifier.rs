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

use rune_ir::{self, RuneIr};

pub fn verify_bytes(data: &[u8]) -> Result<RuneIr, String> {
    let ir = rune_ir::parse_bytes(data)?;
    let errs = rune_ir::verify(&ir);
    if errs.is_empty() {
        Ok(ir)
    } else {
        Err(errs.join("; "))
    }
}

pub fn verify_ir(ir: &RuneIr) -> Vec<String> {
    rune_ir::verify(ir)
}
