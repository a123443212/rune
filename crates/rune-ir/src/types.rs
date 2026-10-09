use serde::{Deserialize, Serialize};

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
    #[serde(default)]
    pub board_size: usize,
    #[serde(default)]
    pub channels: usize,
    #[serde(default)]
    pub num_blocks: usize,
    #[serde(default)]
    pub policy_size: usize,
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
    pub buffers: std::collections::BTreeMap<String, BufferPlace>,
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
    pub hashes: std::collections::BTreeMap<String, String>,
}
