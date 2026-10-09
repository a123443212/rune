pub fn fusion_of(kind: &str) -> String {
    match kind {
        "Q" | "K" | "V" => "f_qkv".to_string(),
        "Score" | "Bias" | "Gate" => "f_score_bias_gate".to_string(),
        "Mix" | "Residual" => "f_mix_residual".to_string(),
        "HeadH1" => "f_head_h1".to_string(),
        "HeadH2" => "f_head_h2".to_string(),
        "StemConv" | "Conv2D" => "f_conv".to_string(),
        "ResidualAdd" | "Relu" => "f_resblock".to_string(),
        "PolicyLogits" | "Policy" => "f_policy".to_string(),
        _ => String::new(),
    }
}
