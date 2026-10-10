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

use std::path::PathBuf;

#[test]
fn resnet_model_compiles() {
    let p = PathBuf::from("../../spec/test-vectors/models/resnet9-fp32.rune");
    let m = rune_model::load(&p).expect("resnet fixture");
    let mut ir = rune_compiler::build_from_model(&m, "portable", "generic-x86-64").expect("ir");
    rune_compiler::select_kernels(&mut ir);
    let errs = rune_ir::verify(&ir);
    assert!(errs.is_empty(), "{:?}", errs);
    let kinds: Vec<&str> = ir.ops.iter().map(|o| o.kind.as_str()).collect();
    assert!(kinds.contains(&"StemConv"));
    assert!(kinds.contains(&"Policy"));
}

#[test]
fn classic_model_still_compiles() {
    let p = PathBuf::from("../../spec/test-vectors/models/tiny-mlp-fp32.rune");
    let m = rune_model::load(&p).expect("fixture");
    let mut ir = rune_compiler::build_from_model(&m, "portable", "generic-x86-64").expect("ir");
    rune_compiler::select_kernels(&mut ir);
    assert!(rune_ir::verify(&ir).is_empty());
}
