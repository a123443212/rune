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
