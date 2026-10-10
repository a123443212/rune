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

use rune_ir::incremental::{sparse_kernel_for, IncrementalMeta};
use rune_ir::{KernelEntry, RuneIr};

pub fn select_incremental_kernels(ir: &RuneIr, meta: &IncrementalMeta) -> Vec<KernelEntry> {
    let tokens = ir.model.tokens;
    let dim = ir.model.token_dim;
    let mut plan = Vec::new();
    for op in &ir.ops {
        let kid = sparse_kernel_for(&op.kind, tokens, dim);
        if kid.is_empty() {
            continue;
        }
        plan.push(KernelEntry {
            op: op.id.clone(),
            kind: op.kind.clone(),
            kernel_id: kid,
            shape: rune_ir::shape_key(tokens, dim, ir.model.head_h1, ir.model.head_h2, &op.kind),
            dtype: ir.model.dtype.clone(),
            packing: meta.cache_layout.clone(),
            isa: meta.isa.clone(),
            fusion_group: "sparse_interaction".to_string(),
        });
    }
    plan
}

pub fn incremental_cache_key(model_hash: &str, arch: &str, precision: &str, isa: &str, compiler: &str, meta: &IncrementalMeta) -> String {
    format!(
        "{}|{}|{}|{}|{}|{}|{}|{}",
        model_hash,
        arch,
        precision,
        isa,
        compiler,
        meta.interaction_graph_version,
        meta.cache_layout,
        meta.invalidation_version
    )
}

pub fn incremental_flops(tokens: usize, dim: usize, changed: usize) -> usize {
    let qkv = changed * dim * dim * 3;
    let cells = 2 * changed * tokens - changed * changed;
    let score = cells * dim;
    let mix = tokens * tokens * dim;
    qkv + score + mix
}

pub fn full_flops(tokens: usize, dim: usize) -> usize {
    incremental_flops(tokens, dim, tokens)
}
