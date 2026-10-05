use serde::{Deserialize, Serialize};

pub const IR_VERSION: &str = "1.0";
pub const SPEC_VERSION: &str = "RUNE-10";
pub const COMPILER_VERSION: &str = "0.11.0";
pub const PACKING_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrModel {
    pub architecture: String,
    pub architecture_version: String,
    pub tokens: usize,
    pub token_dim: usize,
    pub dtype: String,
    pub quantization: String,
    pub gate: String,
    pub alpha: f32,
    pub head_h1: usize,
    pub head_h2: usize,
    pub threshold: f32,
    pub t_high: f32,
    pub t_low: Option<f32>,
    pub has_t_low: bool,
    pub adaptive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub layout: String,
    pub constant: bool,
    pub lifetime: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrOp {
    pub id: String,
    pub kind: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrTarget {
    pub cpu: String,
    pub isa: String,
    pub vector_width: usize,
    pub dtype: String,
    pub quantization: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelEntry {
    pub op: String,
    pub kind: String,
    pub kernel_id: String,
    pub shape: String,
    pub dtype: String,
    pub packing: String,
    pub isa: String,
    pub fusion_group: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionGroup {
    pub id: String,
    pub ops: Vec<String>,
    pub kernel: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferPlace {
    pub offset: usize,
    pub elems: usize,
    pub bytes: usize,
    pub shared: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPlan {
    pub arena_bytes: usize,
    pub alignment: usize,
    pub buffers: std::collections::HashMap<String, BufferPlace>,
    pub strategy: String,
    pub in_place: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuneIr {
    pub ir_version: String,
    pub spec_version: String,
    pub model: IrModel,
    pub tensors: Vec<IrTensor>,
    pub ops: Vec<IrOp>,
    pub memory: MemoryPlan,
    pub fusion: Vec<FusionGroup>,
    pub kernel_plan: Vec<KernelEntry>,
    pub target: IrTarget,
    pub hashes: std::collections::HashMap<String, String>,
}

pub fn valid_isa(isa: &str) -> bool {
    isa == "portable" || isa == "avx2" || isa == "avx512"
}

pub fn verify(ir: &RuneIr) -> Vec<String> {
    let mut errs = Vec::new();
    if ir.ir_version != IR_VERSION {
        errs.push(format!("bad ir_version {}", ir.ir_version));
    }
    let kinds: Vec<&str> = ir.ops.iter().map(|o| o.kind.as_str()).collect();
    for need in ["FeatureUpdate", "AccumulatorUpdate", "Tokenize", "Q", "K", "V", "Score", "Gate", "Mix", "Residual", "HeadH1", "HeadH2", "Value", "WDL"] {
        if !kinds.contains(&need) {
            errs.push(format!("missing op {}", need));
        }
    }
    if !valid_isa(&ir.target.isa) {
        errs.push(format!("unsupported isa {}", ir.target.isa));
    }
    if ir.model.tokens == 0 || ir.model.tokens > 16 {
        errs.push(format!("unsupported tokens {}", ir.model.tokens));
    }
    if ir.model.token_dim == 0 || ir.model.token_dim > 128 {
        errs.push(format!("unsupported dim {}", ir.model.token_dim));
    }
    if ir.model.quantization != "fp32" && ir.model.quantization != "int8" && ir.model.quantization != "int16" {
        errs.push(format!("unsupported quantization {}", ir.model.quantization));
    }
    errs
}

pub fn parse_bytes(data: &[u8]) -> Result<RuneIr, String> {
    serde_json::from_slice(data).map_err(|e| e.to_string())
}

pub fn shape_key(tokens: usize, dim: usize, h1: usize, h2: usize, kind: &str) -> String {
    match kind {
        "Q" | "K" | "V" => format!("{}x{}", tokens, dim),
        "Score" | "Bias" | "Gate" => format!("{}x{}", tokens, tokens),
        "Mix" | "Residual" => format!("{}x{}", tokens, dim),
        "HeadH1" => format!("{}x{}", h1, tokens * dim),
        "HeadH2" => format!("{}x{}", h2, h1),
        "Value" => format!("1x{}", h2),
        "WDL" => format!("3x{}", h2),
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
        _ => "generic".to_string(),
    }
}
