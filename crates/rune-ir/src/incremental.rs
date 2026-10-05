use serde::{Deserialize, Serialize};

pub const GRAPH_VERSION: &str = "v13-graph-01";
pub const INVALIDATION_VERSION: &str = "v13-inv-01";
pub const CACHE_LAYOUT: &str = "v13-cache-rowmajor-aligned32";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalMeta {
    pub incremental_support: bool,
    pub interaction_graph_version: String,
    pub cache_layout: String,
    pub invalidation_version: String,
    pub threshold: usize,
    pub precision: String,
    pub isa: String,
}

impl IncrementalMeta {
    pub fn new(threshold: usize, precision: &str, isa: &str) -> IncrementalMeta {
        IncrementalMeta {
            incremental_support: true,
            interaction_graph_version: GRAPH_VERSION.to_string(),
            cache_layout: CACHE_LAYOUT.to_string(),
            invalidation_version: INVALIDATION_VERSION.to_string(),
            threshold,
            precision: precision.to_string(),
            isa: isa.to_string(),
        }
    }

    pub fn verify(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.interaction_graph_version != GRAPH_VERSION {
            errs.push(format!("graph mismatch {}", self.interaction_graph_version));
        }
        if self.invalidation_version != INVALIDATION_VERSION {
            errs.push(format!("invalidation mismatch {}", self.invalidation_version));
        }
        if self.cache_layout != CACHE_LAYOUT {
            errs.push(format!("cache layout mismatch {}", self.cache_layout));
        }
        if self.precision != "fp32" && self.precision != "int8" {
            errs.push(format!("precision mismatch {}", self.precision));
        }
        errs
    }
}

pub fn sparse_kernel_for(op_kind: &str, tokens: usize, dim: usize) -> String {
    match op_kind {
        "Q" | "K" | "V" => format!("qkv_rows_{}x{}", tokens, dim),
        "Score" | "Bias" | "Gate" => format!("score_update_{}x{}", tokens, tokens),
        "Mix" | "Residual" => format!("mix_update_{}x{}", tokens, dim),
        _ => String::new(),
    }
}

pub fn cache_bytes(tokens: usize, dim: usize) -> (usize, usize, usize) {
    (tokens * tokens * 4, tokens * tokens * 4, tokens * dim * 4)
}
