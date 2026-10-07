use std::collections::BTreeMap;
use rune_ir::{BufferPlace, KernelEntry, MemoryPlan, RuneIr};

fn fusion_of(kind: &str) -> String {
    match kind {
        "Q" | "K" | "V" => "f_qkv".to_string(),
        "Score" | "Bias" | "Gate" => "f_score_bias_gate".to_string(),
        "Mix" | "Residual" => "f_mix_residual".to_string(),
        "HeadH1" => "f_head_h1".to_string(),
        "HeadH2" => "f_head_h2".to_string(),
        _ => String::new(),
    }
}

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
        plan.push(KernelEntry {
            op: op.id.clone(),
            kind: op.kind.clone(),
            kernel_id: kid,
            shape,
            dtype: dtype.clone(),
            packing: "row-major-aligned32".to_string(),
            isa: isa.clone(),
            fusion_group: fusion_of(&op.kind),
        });
    }
    ir.kernel_plan = plan;
    ir.memory = plan_memory(tokens, dim, h1, h2);
}

fn align_up(n: usize, a: usize) -> usize {
    ((n + a - 1) / a) * a
}

pub fn plan_memory(tokens: usize, dim: usize, h1: usize, h2: usize) -> MemoryPlan {
    let live: Vec<(&str, usize)> = vec![
        ("tokens_buf", tokens * dim),
        ("q_buf", tokens * dim),
        ("k_buf", tokens * dim),
        ("v_buf", tokens * dim),
        ("scores_buf", tokens * tokens),
        ("gate_buf", tokens * tokens),
        ("mixed_raw_buf", tokens * dim),
        ("mixed_buf", tokens * dim),
        ("h1_buf", h1),
        ("h2_buf", h2),
        ("flat_buf", tokens * dim),
        ("tmp_row", dim.max(h1).max(h2)),
    ];
    let groups: Vec<Vec<&str>> = vec![
        vec!["q_buf", "h1_buf"],
        vec!["k_buf", "h2_buf"],
        vec!["scores_buf", "gate_buf"],
    ];
    let mut places: BTreeMap<String, BufferPlace> = BTreeMap::new();
    let mut off: usize = 0;
    for (name, elems) in live {
        let mut shared_off: Option<usize> = None;
        for g in &groups {
            if g.contains(&name) {
                for other in g {
                    if *other != name {
                        if let Some(p) = places.get(*other) {
                            if p.elems >= elems {
                                shared_off = Some(p.offset);
                                break;
                            }
                        }
                    }
                }
            }
            if shared_off.is_some() {
                break;
            }
        }
        if let Some(s) = shared_off {
            places.insert(name.to_string(), BufferPlace { offset: s, elems, bytes: elems * 4, shared: true });
        } else {
            let s = align_up(off, 32);
            places.insert(name.to_string(), BufferPlace { offset: s, elems, bytes: elems * 4, shared: false });
            off = s + elems * 4;
        }
    }
    MemoryPlan {
        arena_bytes: align_up(off, 32),
        alignment: 32,
        buffers: places,
        strategy: "reuse q/h1, k/h2, scores/gate; single bump arena, no per-eval alloc".to_string(),
        in_place: vec!["gate_in_scores".to_string(), "mixed_raw_into_mixed_when_alpha_1".to_string()],
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
    if kernel_id.contains("fused") || kernel_id.starts_with("score") || kernel_id.starts_with("mix") || kernel_id.starts_with("qkv") {
        mem *= 0.6;
    }
    let f = if isa == "avx512" { 0.28 } else if isa == "avx2" { 0.35 } else { 1.0 };
    ops * f + mem * 0.05
}
