use std::path::PathBuf;

pub fn run(model: &str, out: &str, isa: &str, cpu: &str) -> i32 {
    let mp = PathBuf::from(model);
    let m = match rune_model::load(&mp) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("load failed: {}", e);
            return 1;
        }
    };
    let mut ir = match rune_compiler::build_from_model(&m, isa, cpu) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("ir failed: {}", e);
            return 1;
        }
    };
    rune_compiler::fuse_plan(&mut ir);
    rune_compiler::select_kernels(&mut ir);
    let errs = rune_ir::verify(&ir);
    if !errs.is_empty() {
        eprintln!("ir invalid: {}", errs.join("; "));
        return 2;
    }
    let kp = serde_json::to_vec(&ir.kernel_plan).unwrap_or_default();
    let mp2 = serde_json::to_vec(&ir.memory).unwrap_or_default();
    let tp = serde_json::to_vec(&ir.target).unwrap_or_default();
    let ph = rune_compiler::plan_hash(&kp, &mp2, &tp);
    let payload = m.payload_bytes.clone();
    let mut header = m.header.raw.clone();
    header["compiled"] = serde_json::Value::Bool(true);
    header["compiled_kind"] = serde_json::Value::String("compiled-v11".to_string());
    header["rune_ir_version"] = serde_json::Value::String(rune_ir::IR_VERSION.to_string());
    header["compiler_version"] = serde_json::Value::String(rune_ir::COMPILER_VERSION.to_string());
    header["target_isa"] = serde_json::Value::String(isa.to_string());
    header["target_cpu"] = serde_json::Value::String(cpu.to_string());
    header["kernel_plan"] = serde_json::to_value(&ir.kernel_plan).unwrap_or(serde_json::Value::Array(vec![]));
    header["fusion_plan"] = serde_json::to_value(&ir.fusion).unwrap_or(serde_json::Value::Array(vec![]));
    header["memory_plan"] = serde_json::to_value(&ir.memory).unwrap_or(serde_json::Value::Null);
    header["kernel_plan_hash"] = serde_json::Value::String(ph.clone());
    header["source_hash"] = serde_json::Value::String(rune_compiler::source_hash(b"{}", &payload));
    let bytes = rune_compiler::write_compiled(header, &payload);
    match std::fs::write(out, &bytes) {
        Ok(()) => {
            println!("compiled {} isa {} bytes {} plan {}", out, isa, bytes.len(), ph);
            0
        }
        Err(e) => {
            eprintln!("write failed: {}", e);
            1
        }
    }
}
