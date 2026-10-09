use super::arch::{arch_family, required_ops_for_arch};
use super::types::RuneIr;
use super::version::{ACCEPTED_IR_VERSIONS, ACCEPTED_SPEC_VERSIONS};

pub fn valid_isa(isa: &str) -> bool {
    isa == "portable" || isa == "avx2" || isa == "avx512"
}

pub fn verify(ir: &RuneIr) -> Vec<String> {
    let mut errs = Vec::new();
    if !ACCEPTED_IR_VERSIONS.contains(&ir.ir_version.as_str()) {
        errs.push(format!("bad ir_version {}", ir.ir_version));
    }
    if !ACCEPTED_SPEC_VERSIONS.contains(&ir.spec_version.as_str()) {
        errs.push(format!("bad spec_version {}", ir.spec_version));
    }
    let kinds: Vec<&str> = ir.ops.iter().map(|o| o.kind.as_str()).collect();
    let family = arch_family(&ir.model.architecture);
    if family == "unknown" {
        errs.push(format!("unsupported architecture {}", ir.model.architecture));
        return errs;
    }
    for need in required_ops_for_arch(&ir.model.architecture) {
        if !kinds.contains(&need) {
            errs.push(format!("missing op {} for {}", need, ir.model.architecture));
        }
    }
    let ids: Vec<&str> = ir.ops.iter().map(|o| o.id.as_str()).collect();
    for f in &ir.fusion {
        for op in &f.ops {
            if !kinds.contains(&op.as_str()) {
                errs.push(format!("fusion {} references missing op {}", f.id, op));
            }
        }
        if f.kernel.is_empty() {
            errs.push(format!("fusion {} missing kernel", f.id));
        }
    }
    for k in &ir.kernel_plan {
        if k.kernel_id.is_empty() {
            errs.push(format!("kernel entry {} missing kernel", k.op));
        }
        if !ids.contains(&k.op.as_str()) {
            errs.push(format!("kernel entry {} references missing op {}", k.kernel_id, k.op));
        }
    }
    if !valid_isa(&ir.target.isa) {
        errs.push(format!("unsupported isa {}", ir.target.isa));
    }
    if family == "resnet" {
        if ir.model.board_size == 0 || ir.model.board_size > 19 {
            errs.push(format!("unsupported board_size {}", ir.model.board_size));
        }
        if ir.model.channels == 0 || ir.model.channels > 256 {
            errs.push(format!("unsupported channels {}", ir.model.channels));
        }
        if ir.model.num_blocks > 64 {
            errs.push(format!("unsupported num_blocks {}", ir.model.num_blocks));
        }
        if ir.model.policy_size > 512 {
            errs.push(format!("unsupported policy_size {}", ir.model.policy_size));
        }
    } else {
        if ir.model.tokens == 0 || ir.model.tokens > 16 {
            errs.push(format!("unsupported tokens {}", ir.model.tokens));
        }
        if ir.model.token_dim == 0 || ir.model.token_dim > 128 {
            errs.push(format!("unsupported dim {}", ir.model.token_dim));
        }
    }
    if ir.model.quantization != "fp32" && ir.model.quantization != "int8" && ir.model.quantization != "int16" {
        errs.push(format!("unsupported quantization {}", ir.model.quantization));
    }
    errs
}

pub fn parse_bytes(data: &[u8]) -> Result<RuneIr, String> {
    serde_json::from_slice(data).map_err(|e| e.to_string())
}
