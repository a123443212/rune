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

use rune_ir::{KernelEntry, RuneIr};
use super::planner_fusion::fusion_of;
use super::planner_memory::{plan_memory, plan_memory_resnet};

pub fn select_kernels(ir: &mut RuneIr) {
    let tokens = ir.model.tokens;
    let dim = ir.model.token_dim;
    let h1 = ir.model.head_h1;
    let h2 = ir.model.head_h2;
    let isa = ir.target.isa.clone();
    let dtype = ir.model.dtype.clone();
    let mut plan: Vec<KernelEntry> = Vec::new();
    for op in &ir.ops {
        let shape = rune_ir::shape_key(tokens, dim, h1, h2, &op.kind);
        let mut kid = rune_ir::kernel_for(&op.kind, &shape);
        if op.kind == "Tokenize" && (ir.model.quantization == "int8" || ir.model.quantization == "int16") {
            kid = "dequant_clip".to_string();
        }
        plan.push(KernelEntry { op: op.id.clone(), kind: op.kind.clone(), kernel_id: kid, shape, dtype: dtype.clone(), packing: "row-major-aligned32".to_string(), isa: isa.clone(), fusion_group: fusion_of(&op.kind) });
    }
    ir.kernel_plan = plan;
    if rune_ir::is_resnet_arch(&ir.model.architecture) {
        ir.memory = plan_memory_resnet(ir.model.board_size, ir.model.channels, ir.model.policy_size, h2);
    } else {
        ir.memory = plan_memory(tokens, dim, h1, h2);
    }
}

pub fn kernel_cost(shape: &str, kernel_id: &str, isa: &str) -> f64 {
    let mut ops = 1.0;
    let parts: Vec<&str> = shape.split('x').collect();
    if parts.len() == 2 {
        if let (Ok(a), Ok(b)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
            ops = a * b;
        }
    }
    let mut mem = ops * 4.0;
    if kernel_id.contains("fused") || kernel_id.starts_with("score") || kernel_id.starts_with("mix") || kernel_id.starts_with("qkv") || kernel_id.starts_with("conv") {
        mem *= 0.6;
    }
    let f = if isa == "avx512" { 0.28 } else if isa == "avx2" { 0.35 } else { 1.0 };
    ops * f + mem * 0.05
}
