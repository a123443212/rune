use std::collections::BTreeMap;
use rune_ir::{FusionGroup, IrModel, IrOp, IrTarget, IrTensor, MemoryPlan, RuneIr};
use rune_model::RuneModel;
use serde_json::json;

fn tensor_entry(name: &str, shape: Vec<usize>, dtype: &str, constant: bool, life: Vec<usize>) -> IrTensor {
    IrTensor {
        name: name.to_string(),
        shape,
        dtype: dtype.to_string(),
        layout: "row-major".to_string(),
        constant,
        lifetime: life,
    }
}

fn op_entry(id: &str, kind: &str, inputs: Vec<&str>, outputs: Vec<&str>, attrs: serde_json::Value) -> IrOp {
    IrOp {
        id: id.to_string(),
        kind: kind.to_string(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs: outputs.iter().map(|s| s.to_string()).collect(),
        attrs,
    }
}

fn empty_memory() -> MemoryPlan {
    MemoryPlan {
        arena_bytes: 0,
        alignment: 32,
        buffers: BTreeMap::new(),
        strategy: String::new(),
        in_place: Vec::new(),
    }
}

pub fn build_from_model(m: &RuneModel, isa: &str, cpu: &str) -> Result<RuneIr, String> {
    let arch = m.header.architecture_id.clone();
    let tokens = m.header.tokens;
    let dim = m.header.token_dim;
    let quant = m.header.quantization.clone();
    let raw = &m.header.raw;
    let gate = raw.get("gate").and_then(|v| v.as_str()).unwrap_or("clip").to_string();
    let alpha = raw.get("alpha").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let h1 = raw.get("head_h1").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
    let h2 = raw.get("head_h2").and_then(|v| v.as_u64()).unwrap_or(32) as usize;
    let thr = raw.get("threshold").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
    let th = raw.get("t_high").and_then(|v| v.as_f64()).unwrap_or(thr as f64) as f32;
    let tl = raw.get("t_low").and_then(|v| v.as_f64()).map(|v| v as f32);
    if tokens == 0 || tokens > 16 {
        return Err(format!("unsupported tokens {}", tokens));
    }
    if dim == 0 || dim > 128 {
        return Err(format!("unsupported dim {}", dim));
    }
    if isa != "portable" && isa != "avx2" && isa != "avx512" {
        return Err(format!("unsupported isa {}", isa));
    }
    if quant != "fp32" && quant != "int8" && quant != "int16" {
        return Err(format!("unsupported quantization {}", quant));
    }
    if gate != "clip" && gate != "hard_sigmoid" {
        return Err(format!("unsupported gate {}", gate));
    }
    if h1 == 0 || h1 > 4096 || h2 == 0 || h2 > 4096 {
        return Err(format!("unsupported head {}x{}", h1, h2));
    }
    let adaptive = arch == "RUNE-04" || arch == "RUNE-05";
    let model = IrModel {
        architecture: arch,
        architecture_version: m.header.architecture_version.clone(),
        tokens,
        token_dim: dim,
        dtype: if quant == "fp32" { "fp32".to_string() } else { quant.clone() },
        quantization: quant.clone(),
        gate,
        alpha,
        head_h1: h1,
        head_h2: h2,
        threshold: thr,
        t_high: th,
        t_low: tl,
        has_t_low: tl.is_some(),
        adaptive,
    };
    let vw = if isa == "avx512" { 16 } else if isa == "avx2" { 8 } else { 1 };
    let target = IrTarget {
        cpu: cpu.to_string(),
        isa: isa.to_string(),
        vector_width: vw,
        dtype: model.dtype.clone(),
        quantization: quant.clone(),
    };
    let mut tensors: Vec<IrTensor> = Vec::new();
    tensors.push(tensor_entry("features", vec![tokens], "index", false, vec![0, 1]));
    tensors.push(tensor_entry("accumulator", vec![tokens, dim], "acc", false, vec![0, 2]));
    tensors.push(tensor_entry("tokens", vec![tokens, dim], "fp32", false, vec![1, 3]));
    for tm in &m.header.tensors {
        let is_const = !tm.name.starts_with("features");
        let life = vec![0, 12];
        tensors.push(tensor_entry(&tm.name, tm.shape.clone(), &tm.dtype, is_const, life));
    }
    let mut ops: Vec<IrOp> = Vec::new();
    ops.push(op_entry("op00", "FeatureUpdate", vec![], vec!["features"], json!({"tokens": tokens})));
    ops.push(op_entry("op01", "AccumulatorUpdate", vec!["features"], vec!["accumulator"], json!({"tokens": tokens, "dim": dim, "quant": quant})));
    ops.push(op_entry("op02", "Tokenize", vec!["accumulator"], vec!["tokens"], json!({"tokens": tokens, "dim": dim})));
    ops.push(op_entry("op03", "Q", vec!["tokens", "wq", "bq"], vec!["Q"], json!({"tokens": tokens, "dim": dim})));
    ops.push(op_entry("op04", "K", vec!["tokens", "wk", "bk"], vec!["K"], json!({"tokens": tokens, "dim": dim})));
    ops.push(op_entry("op05", "V", vec!["tokens", "wvv", "bvv"], vec!["V"], json!({"tokens": tokens, "dim": dim})));
    ops.push(op_entry("op06", "Score", vec!["Q", "K"], vec!["scores"], json!({"tokens": tokens, "dim": dim})));
    ops.push(op_entry("op07", "Bias", vec!["scores", "gabS"], vec!["biased"], json!({"tokens": tokens})));
    ops.push(op_entry("op08", "Gate", vec!["biased"], vec!["gate"], json!({"tokens": tokens, "gate": model.gate})));
    ops.push(op_entry("op09", "Mix", vec!["gate", "V"], vec!["mixed_raw"], json!({"tokens": tokens, "dim": dim})));
    ops.push(op_entry("op10", "Residual", vec!["tokens", "mixed_raw"], vec!["mixed"], json!({"tokens": tokens, "dim": dim, "alpha": alpha})));
    ops.push(op_entry("op11", "HeadH1", vec!["mixed"], vec!["h1"], json!({"input": tokens * dim, "output": h1})));
    ops.push(op_entry("op12", "HeadH2", vec!["h1"], vec!["h2"], json!({"input": h1, "output": h2})));
    ops.push(op_entry("op13", "Value", vec!["h2", "wvo", "bvo"], vec!["value"], json!({"input": h2})));
    ops.push(op_entry("op14", "WDL", vec!["h2", "wwdl", "bwdl"], vec!["wdl"], json!({"input": h2, "output": 3})));
    if adaptive {
        ops.push(op_entry("op15", "Route", vec!["h1"], vec!["route"], json!({"threshold": thr, "t_high": th})));
    }
    let fusion: Vec<FusionGroup> = Vec::new();
    let ir = RuneIr {
        ir_version: rune_ir::IR_VERSION.to_string(),
        spec_version: rune_ir::SPEC_VERSION.to_string(),
        model,
        tensors,
        ops,
        memory: empty_memory(),
        fusion,
        kernel_plan: Vec::new(),
        target,
        hashes: BTreeMap::new(),
    };
    Ok(ir)
}
