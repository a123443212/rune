use rune_ir::{self, RuneIr};

pub fn verify_bytes(data: &[u8]) -> Result<RuneIr, String> {
    let ir = rune_ir::parse_bytes(data)?;
    let errs = rune_ir::verify(&ir);
    if errs.is_empty() {
        Ok(ir)
    } else {
        Err(errs.join("; "))
    }
}

pub fn verify_ir(ir: &RuneIr) -> Vec<String> {
    rune_ir::verify(ir)
}
