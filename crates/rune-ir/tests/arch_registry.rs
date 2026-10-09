use rune_ir::{arch_family, is_resnet_arch, required_ops_for_arch};

#[test]
fn classic_family_detected() {
    assert_eq!(arch_family("RUNE-ATTN-GAB"), "classic");
    assert_eq!(arch_family("RUNE-MLP"), "classic");
    assert!(!is_resnet_arch("RUNE-MLP"));
}

#[test]
fn resnet_family_detected() {
    assert_eq!(arch_family("RUNE-RESNET-01"), "resnet");
    assert!(is_resnet_arch("RUNE-RESNET-01"));
    let ops = required_ops_for_arch("RUNE-RESNET-01");
    assert!(ops.contains(&"StemConv"));
    assert!(ops.contains(&"Policy"));
}

#[test]
fn unknown_family_empty_ops() {
    assert_eq!(arch_family("RUNE-XYZ"), "unknown");
    assert!(required_ops_for_arch("RUNE-XYZ").is_empty());
}

#[test]
fn resnet_ir_verifies() {
    let ir = rune_ir::RuneIr {
        ir_version: "1.1".to_string(),
        spec_version: "RUNE-11".to_string(),
        model: rune_ir::IrModel { architecture: "RUNE-RESNET-01".to_string(), architecture_version: "0.1.0".to_string(), tokens: 9, token_dim: 16, dtype: "fp32".to_string(), quantization: "fp32".to_string(), gate: "relu".to_string(), alpha: 1.0, head_h1: 1296, head_h2: 32, threshold: 0.5, t_high: 0.5, t_low: None, has_t_low: false, adaptive: false, board_size: 9, channels: 16, num_blocks: 1, policy_size: 82 },
        tensors: Vec::new(),
        ops: ["StemConv", "Conv2D", "ResidualAdd", "Relu", "GlobalPool", "Flatten", "Value", "WDL", "PolicyLogits", "Policy"].iter().map(|k| rune_ir::IrOp { id: k.to_string(), kind: k.to_string(), inputs: Vec::new(), outputs: Vec::new(), attrs: serde_json::json!({}) }).collect(),
        memory: rune_ir::MemoryPlan { arena_bytes: 0, alignment: 32, buffers: Default::default(), strategy: String::new(), in_place: Vec::new() },
        fusion: Vec::new(),
        kernel_plan: Vec::new(),
        target: rune_ir::IrTarget { cpu: "x".to_string(), isa: "portable".to_string(), vector_width: 1, dtype: "fp32".to_string(), quantization: "fp32".to_string() },
        hashes: Default::default(),
    };
    assert!(rune_ir::verify(&ir).is_empty());
}
