use std::collections::BTreeMap;
use rune_ir::{IrOp, IrTensor, MemoryPlan};
use serde_json::json;

pub fn tensor_entry(name: &str, shape: Vec<usize>, dtype: &str, constant: bool, life: Vec<usize>) -> IrTensor {
    IrTensor { name: name.to_string(), shape, dtype: dtype.to_string(), layout: "row-major".to_string(), constant, lifetime: life }
}

pub fn op_entry(id: &str, kind: &str, inputs: Vec<&str>, outputs: Vec<&str>, attrs: serde_json::Value) -> IrOp {
    IrOp { id: id.to_string(), kind: kind.to_string(), inputs: inputs.iter().map(|s| s.to_string()).collect(), outputs: outputs.iter().map(|s| s.to_string()).collect(), attrs }
}

pub fn empty_memory() -> MemoryPlan {
    MemoryPlan { arena_bytes: 0, alignment: 32, buffers: BTreeMap::new(), strategy: String::new(), in_place: Vec::new() }
}

pub fn check_isa_quant(isa: &str, quant: &str) -> Result<(), String> {
    if isa != "portable" && isa != "avx2" && isa != "avx512" {
        return Err(format!("unsupported isa {}", isa));
    }
    if quant != "fp32" && quant != "int8" && quant != "int16" {
        return Err(format!("unsupported quantization {}", quant));
    }
    Ok(())
}

pub fn vector_width(isa: &str) -> usize {
    if isa == "avx512" { 16 } else if isa == "avx2" { 8 } else { 1 }
}

pub fn json_wrap() -> serde_json::Value {
    json!({})
}
