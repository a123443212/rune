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

use rune_ir::{IrModel, IrTarget};
use rune_model::RuneModel;
use serde_json::json;
use super::parser_common::{check_isa_quant, empty_memory, op_entry, tensor_entry, vector_width};
use std::collections::BTreeMap;

pub fn build_resnet(m: &RuneModel, isa: &str, cpu: &str) -> Result<rune_ir::RuneIr, String> {
    let arch = m.header.architecture_id.clone();
    let quant = m.header.quantization.clone();
    let raw = &m.header.raw;
    check_isa_quant(isa, &quant)?;
    let board = raw.get("board_size").and_then(|v| v.as_u64()).unwrap_or(19) as usize;
    let channels = raw.get("channels").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
    let blocks = raw.get("num_blocks").and_then(|v| v.as_u64()).unwrap_or(6) as usize;
    let policy = raw.get("policy_size").and_then(|v| v.as_u64()).unwrap_or((board * board + 1) as u64) as usize;
    let h2 = raw.get("head_h2").and_then(|v| v.as_u64()).unwrap_or(256) as usize;
    let in_planes = raw.get("in_planes").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    if board == 0 || board > 19 {
        return Err(format!("unsupported board_size {}", board));
    }
    if channels == 0 || channels > 256 {
        return Err(format!("unsupported channels {}", channels));
    }
    if blocks > 64 {
        return Err(format!("unsupported num_blocks {}", blocks));
    }
    if in_planes == 0 || in_planes > 16 {
        return Err(format!("unsupported in_planes {}", in_planes));
    }
    let tokens = board;
    let dim = channels;
    let model = IrModel { architecture: arch, architecture_version: m.header.architecture_version.clone(), tokens, token_dim: dim, dtype: if quant == "fp32" { "fp32".to_string() } else { quant.clone() }, quantization: quant.clone(), gate: "relu".to_string(), alpha: 1.0, head_h1: channels * board * board, head_h2: h2, threshold: 0.5, t_high: 0.5, t_low: None, has_t_low: false, adaptive: false, board_size: board, channels, num_blocks: blocks, policy_size: policy, in_planes };
    let vw = vector_width(isa);
    let target = IrTarget { cpu: cpu.to_string(), isa: isa.to_string(), vector_width: vw, dtype: model.dtype.clone(), quantization: quant.clone() };
    let mut tensors = Vec::new();
    tensors.push(tensor_entry("planes", vec![in_planes * board * board], "index", false, vec![0, 1]));
    tensors.push(tensor_entry("stem", vec![board, board], "fp32", false, vec![1, 3]));
    for tm in &m.header.tensors {
        let is_const = !tm.name.starts_with("planes");
        tensors.push(tensor_entry(&tm.name, tm.shape.clone(), &tm.dtype, is_const, vec![0, 12]));
    }
    let mut ops = Vec::new();
    ops.push(op_entry("op00", "FeaturePlanes", vec![], vec!["planes"], json!({"board": board, "in_planes": in_planes})));
    ops.push(op_entry("op01", "StemConv", vec!["planes", "stem_w", "stem_b"], vec!["x0"], json!({"board": board, "channels": channels, "in_planes": in_planes})));
    ops.push(op_entry("op02", "Relu", vec!["x0"], vec!["x1"], json!({})));
    let mut cur = String::from("x1");
    let mut nid = 3usize;
    for b in 0..blocks {
        let c1 = format!("b{}_c1", b);
        let r1 = format!("b{}_r1", b);
        let c2 = format!("b{}_c2", b);
        let ad = format!("b{}_ad", b);
        let nx = format!("x{}", b + 2);
        ops.push(op_entry(&format!("op{:02}c1", nid), "Conv2D", vec![cur.as_str()], vec![c1.as_str()], json!({"board": board, "channels": channels, "block": b})));
        nid += 1;
        ops.push(op_entry(&format!("op{:02}r1", nid), "Relu", vec![c1.as_str()], vec![r1.as_str()], json!({})));
        nid += 1;
        ops.push(op_entry(&format!("op{:02}c2", nid), "Conv2D", vec![r1.as_str()], vec![c2.as_str()], json!({"board": board, "channels": channels, "block": b})));
        nid += 1;
        ops.push(op_entry(&format!("op{:02}ad", nid), "ResidualAdd", vec![cur.as_str(), c2.as_str()], vec![ad.as_str()], json!({})));
        nid += 1;
        ops.push(op_entry(&format!("op{:02}rl", nid), "Relu", vec![ad.as_str()], vec![nx.as_str()], json!({})));
        nid += 1;
        cur = nx;
    }
    ops.push(op_entry("op90", "GlobalPool", vec![cur.as_str()], vec!["pooled"], json!({"board": board, "channels": channels})));
    ops.push(op_entry("op91", "Flatten", vec![cur.as_str()], vec!["flat"], json!({"board": board, "channels": channels})));
    ops.push(op_entry("op92", "Value", vec!["pooled", "wv", "bv"], vec!["value"], json!({"input": channels})));
    ops.push(op_entry("op93", "WDL", vec!["pooled", "wwdl", "bwdl"], vec!["wdl"], json!({"input": channels, "output": 3})));
    ops.push(op_entry("op94", "PolicyLogits", vec!["flat", "wpol", "bpol"], vec!["plogits"], json!({"input": channels * board * board, "output": policy})));
    ops.push(op_entry("op95", "Policy", vec!["plogits"], vec!["policy"], json!({"output": policy})));
    Ok(rune_ir::RuneIr { ir_version: rune_ir::IR_VERSION.to_string(), spec_version: rune_ir::SPEC_VERSION.to_string(), model, tensors, ops, memory: empty_memory(), fusion: Vec::new(), kernel_plan: Vec::new(), target, hashes: BTreeMap::new() })
}
