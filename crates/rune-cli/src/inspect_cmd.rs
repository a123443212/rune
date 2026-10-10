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

pub fn run(model: &str) -> i32 {
    let p = PathBuf::from(model);
    let data = match std::fs::read(&p) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("read failed: {}", e);
            return 1;
        }
    };
    let (header, off) = match rune_compiler::read_header(&data) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("header failed: {}", e);
            return 1;
        }
    };
    let get = |k: &str| header.get(k).cloned().unwrap_or(serde_json::Value::Null);
    println!("bytes {}", data.len());
    println!("payload_off {}", off);
    println!("architecture_id {}", get("architecture_id"));
    println!("target_isa {}", get("target_isa"));
    println!("compiler {}", get("compiler_version"));
    println!("ir {}", get("rune_ir_version"));
    println!("source_hash {}", get("source_hash"));
    println!("plan_hash {}", get("kernel_plan_hash"));
    println!("model_hash {}", get("model_hash"));
    if let Some(plan) = header.get("kernel_plan").and_then(|v| v.as_array()) {
        println!("kernels {}", plan.len());
        for k in plan {
            println!("kernel {} {} {} {}", k.get("kind").and_then(|v| v.as_str()).unwrap_or(""), k.get("kernel_id").and_then(|v| v.as_str()).unwrap_or(""), k.get("shape").and_then(|v| v.as_str()).unwrap_or(""), k.get("isa").and_then(|v| v.as_str()).unwrap_or(""));
        }
    }
    if let Some(mem) = header.get("memory_plan") {
        println!("memory {}", mem);
    }
    0
}
