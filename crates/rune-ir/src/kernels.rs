pub fn shape_key(tokens: usize, dim: usize, h1: usize, h2: usize, kind: &str) -> String {
    match kind {
        "Q" | "K" | "V" => format!("{}x{}", tokens, dim),
        "Score" | "Bias" | "Gate" => format!("{}x{}", tokens, tokens),
        "Mix" | "Residual" => format!("{}x{}", tokens, dim),
        "HeadH1" => format!("{}x{}", h1, tokens * dim),
        "HeadH2" => format!("{}x{}", h2, h1),
        "Value" => format!("1x{}", h2),
        "WDL" => format!("3x{}", h2),
        "StemConv" | "Conv2D" => format!("{}x{}", tokens, dim),
        "ResidualAdd" | "Relu" => format!("{}x{}", tokens, dim),
        "GlobalPool" | "Flatten" => format!("{}x{}", tokens, dim),
        "FeaturePlanes" => format!("{}x{}", tokens, dim),
        "PolicyLogits" | "Policy" => format!("{}x{}", h2, h1),
        _ => format!("{}x{}", tokens, dim),
    }
}

pub fn kernel_for(kind: &str, shape: &str) -> String {
    match kind {
        "Q" | "K" | "V" => {
            if shape == "8x32" {
                "qkv_fused_8x32".to_string()
            } else {
                "matvec_generic".to_string()
            }
        }
        "Score" | "Bias" | "Gate" => {
            if shape == "8x8" {
                "score_bias_gate_8x8".to_string()
            } else if kind == "Score" {
                "matmul_tt_generic".to_string()
            } else if kind == "Bias" {
                "bias_add_generic".to_string()
            } else {
                "gate_generic".to_string()
            }
        }
        "Mix" | "Residual" => {
            if shape == "8x32" {
                "mix_residual_8x32".to_string()
            } else if kind == "Mix" {
                "matmul_generic".to_string()
            } else {
                "residual_generic".to_string()
            }
        }
        "HeadH1" | "HeadH2" => "linear_bias_clip".to_string(),
        "Value" => "dot_tanh".to_string(),
        "WDL" => "matvec_generic".to_string(),
        "FeatureUpdate" => "feature_pack".to_string(),
        "AccumulatorUpdate" => "accum_grouped".to_string(),
        "Tokenize" => "clip".to_string(),
        "Route" => "route_compare".to_string(),
        "StemConv" => "conv2d_3x3".to_string(),
        "Conv2D" => "conv2d_3x3".to_string(),
        "ResidualAdd" => "residual_add".to_string(),
        "Relu" => "relu".to_string(),
        "GlobalPool" => "global_avg_pool".to_string(),
        "Flatten" => "flatten".to_string(),
        "FeaturePlanes" => "feature_planes".to_string(),
        "PolicyLogits" => "matvec_generic".to_string(),
        "Policy" => "softmax".to_string(),
        _ => "generic".to_string(),
    }
}
