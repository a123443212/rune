use std::collections::HashMap;
use std::path::Path;
use rune_ir::RuneIr;
use crate::error::{Result, RuntimeError};

#[derive(Debug, Clone)]
pub struct CompiledHeader {
    pub compiled: bool,
    pub ir_version: String,
    pub compiler_version: String,
    pub target_isa: String,
    pub target_cpu: String,
    pub source_hash: String,
    pub plan_hash: String,
    pub model_hash: String,
    pub kernel_plan: Vec<rune_ir::KernelEntry>,
    pub memory_bytes: usize,
    pub raw: serde_json::Value,
}

fn get_str(v: &serde_json::Value, k: &str) -> String {
    v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string()
}

const MAX_ARENA_BYTES: usize = 1 << 30;

pub fn load_compiled_header(path: &Path) -> Result<CompiledHeader> {
    let data = std::fs::read(path).map_err(|e| RuntimeError::InvalidState(e.to_string()))?;
    let (header, _) = rune_compiler::read_header(&data).map_err(RuntimeError::InvalidState)?;
    let compiled = header.get("compiled").and_then(|v| v.as_bool()).unwrap_or(false);
    if !compiled {
        return Err(RuntimeError::InvalidState("not a compiled artifact".to_string()));
    }
    let irv = get_str(&header, "rune_ir_version");
    if irv != rune_ir::IR_VERSION {
        return Err(RuntimeError::InvalidState(format!("ir version mismatch {}", irv)));
    }
    let isa = get_str(&header, "target_isa");
    if isa != "portable" && isa != "avx2" && isa != "avx512" {
        return Err(RuntimeError::InvalidState(format!("unsupported isa {}", isa)));
    }
    if isa == "avx2" && !rune_kernel::simd::has_avx2_fma() {
        return Err(RuntimeError::InvalidState("avx2 artifact on non-avx2 cpu".to_string()));
    }
    if isa == "avx512" && !rune_kernel::simd::has_avx512() {
        return Err(RuntimeError::InvalidState("avx512 artifact on non-avx512 cpu".to_string()));
    }
    let plan: Vec<rune_ir::KernelEntry> = header
        .get("kernel_plan")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    if plan.is_empty() {
        return Err(RuntimeError::InvalidState("missing kernel plan".to_string()));
    }
    let mem = header.get("memory_plan").and_then(|v| v.get("arena_bytes")).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    if mem > MAX_ARENA_BYTES {
        return Err(RuntimeError::InvalidState("arena bytes out of range".to_string()));
    }
    if let Some(target) = header.get("target") {
        let kp = serde_json::to_vec(&header.get("kernel_plan").cloned().unwrap_or(serde_json::Value::Array(vec![]))).unwrap_or_default();
        let mp = serde_json::to_vec(&header.get("memory_plan").cloned().unwrap_or(serde_json::Value::Null)).unwrap_or_default();
        let tp = serde_json::to_vec(target).unwrap_or_default();
        let want = rune_compiler::plan_hash(&kp, &mp, &tp);
        let got = get_str(&header, "kernel_plan_hash");
        if got != want {
            return Err(RuntimeError::InvalidState("kernel plan hash mismatch".to_string()));
        }
    }
    Ok(CompiledHeader {
        compiled,
        ir_version: irv,
        compiler_version: get_str(&header, "compiler_version"),
        target_isa: isa,
        target_cpu: get_str(&header, "target_cpu"),
        source_hash: get_str(&header, "source_hash"),
        plan_hash: get_str(&header, "kernel_plan_hash"),
        model_hash: get_str(&header, "model_hash"),
        kernel_plan: plan,
        memory_bytes: mem,
        raw: header,
    })
}

pub fn verify_source_hash(path: &Path, expect: &str) -> Result<()> {
    let h = load_compiled_header(path)?;
    if h.source_hash != expect && !expect.is_empty() {
        return Err(RuntimeError::InvalidState("source hash mismatch".to_string()));
    }
    Ok(())
}

pub fn ir_from_header(header: &serde_json::Value) -> Result<RuneIr> {
    let model_v = header.get("model").cloned().unwrap_or(serde_json::Value::Null);
    let _ = model_v;
    let doc = serde_json::json!({
        "ir_version": header.get("rune_ir_version").cloned().unwrap_or(serde_json::Value::String("1.0".to_string())),
        "spec_version": "RUNE-10",
        "model": {
            "architecture": header.get("architecture_id").and_then(|v| v.as_str()).unwrap_or(""),
            "architecture_version": header.get("architecture_version").and_then(|v| v.as_str()).unwrap_or(""),
            "tokens": header.get("tokens").and_then(|v| v.as_u64()).unwrap_or(8),
            "token_dim": header.get("token_dim").and_then(|v| v.as_u64()).unwrap_or(32),
            "dtype": "fp32",
            "quantization": header.get("quantization").and_then(|v| v.as_str()).unwrap_or("fp32"),
            "gate": header.get("gate").and_then(|v| v.as_str()).unwrap_or("clip"),
            "alpha": header.get("alpha").and_then(|v| v.as_f64()).unwrap_or(1.0),
            "head_h1": header.get("head_h1").and_then(|v| v.as_u64()).unwrap_or(128),
            "head_h2": header.get("head_h2").and_then(|v| v.as_u64()).unwrap_or(32),
            "threshold": header.get("threshold").and_then(|v| v.as_f64()).unwrap_or(0.5),
            "t_high": header.get("t_high").and_then(|v| v.as_f64()).unwrap_or(0.5),
            "t_low": header.get("t_low").cloned().unwrap_or(serde_json::Value::Null),
            "has_t_low": header.get("t_low").is_some(),
            "adaptive": false
        },
        "tensors": [],
        "ops": [],
        "memory": header.get("memory_plan").cloned().unwrap_or(serde_json::json!({"arena_bytes": 0, "alignment": 32, "buffers": {}, "strategy": "", "in_place": []})),
        "fusion": header.get("fusion_plan").cloned().unwrap_or(serde_json::Value::Array(vec![])),
        "kernel_plan": header.get("kernel_plan").cloned().unwrap_or(serde_json::Value::Array(vec![])),
        "target": {
            "cpu": header.get("target_cpu").and_then(|v| v.as_str()).unwrap_or("generic-x86-64"),
            "isa": header.get("target_isa").and_then(|v| v.as_str()).unwrap_or("portable"),
            "vector_width": 1,
            "dtype": "fp32",
            "quantization": header.get("quantization").and_then(|v| v.as_str()).unwrap_or("fp32")
        },
        "hashes": {}
    });
    let ir: RuneIr = serde_json::from_value(doc).map_err(|e| RuntimeError::InvalidState(e.to_string()))?;
    let _ = HashMap::<String, String>::new();
    Ok(ir)
}
