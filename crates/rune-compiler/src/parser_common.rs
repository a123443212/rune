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

use std::collections::BTreeMap;
use rune_ir::{IrOp, IrTensor, MemoryPlan};
use serde_json::json;

pub fn tensor_entry(name: &str, shape: Vec<usize>, dtype: &str, constant: bool, life: Vec<usize>) -> IrTensor {
    IrTensor { name: name.to_string(), shape, dtype: dtype.to_string(), layout: "row-major".to_string(), constant, lifetime: life }
}

pub fn op_entry(id: &str, kind: &str, inputs: Vec<&str>, outputs: Vec<&str>, attrs: serde_json::Value) -> IrOp {
    IrOp { id: id.to_string(), kind: kind.to_string(), inputs: inputs.iter().map(|s| s.to_string()).collect(), outputs: outputs.iter().map(|s| s.to_string()).collect(), attrs }
}

pub fn empty_memory() -> MemoryPlan {
    MemoryPlan { arena_bytes: 0, alignment: 32, buffers: BTreeMap::new(), strategy: String::new(), in_place: Vec::new() }
}

pub fn check_isa_quant(isa: &str, quant: &str) -> Result<(), String> {
    if isa != "portable" && isa != "avx2" && isa != "avx512" {
        return Err(format!("unsupported isa {}", isa));
    }
    if quant != "fp32" && quant != "int8" && quant != "int16" {
        return Err(format!("unsupported quantization {}", quant));
    }
    Ok(())
}

pub fn vector_width(isa: &str) -> usize {
    if isa == "avx512" { 16 } else if isa == "avx2" { 8 } else { 1 }
}

pub fn json_wrap() -> serde_json::Value {
    json!({})
}
