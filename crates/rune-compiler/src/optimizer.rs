use rune_ir::{FusionGroup, RuneIr};

pub fn fuse_plan(ir: &mut RuneIr) {
    let mut fusion: Vec<FusionGroup> = Vec::new();
    let kinds: Vec<String> = ir.ops.iter().map(|o| o.kind.clone()).collect();
    let has = |k: &str| kinds.iter().any(|x| x == k);
    if has("Q") && has("K") && has("V") {
        fusion.push(FusionGroup {
            id: "f_qkv".to_string(),
            ops: vec!["Q".to_string(), "K".to_string(), "V".to_string()],
            kernel: "qkv_fused_8x32".to_string(),
        });
    }
    if has("Score") && has("Bias") && has("Gate") {
        fusion.push(FusionGroup {
            id: "f_score_bias_gate".to_string(),
            ops: vec!["Score".to_string(), "Bias".to_string(), "Gate".to_string()],
            kernel: "score_bias_gate_8x8".to_string(),
        });
    }
    if has("Mix") && has("Residual") {
        fusion.push(FusionGroup {
            id: "f_mix_residual".to_string(),
            ops: vec!["Mix".to_string(), "Residual".to_string()],
            kernel: "mix_residual_8x32".to_string(),
        });
    }
    if has("HeadH1") {
        fusion.push(FusionGroup {
            id: "f_head_h1".to_string(),
            ops: vec!["HeadH1".to_string()],
            kernel: "linear_bias_clip".to_string(),
        });
    }
    if has("HeadH2") {
        fusion.push(FusionGroup {
            id: "f_head_h2".to_string(),
            ops: vec!["HeadH2".to_string()],
            kernel: "linear_bias_clip".to_string(),
        });
    }
    ir.fusion = fusion;
}
