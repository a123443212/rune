use std::path::PathBuf;

#[test]
fn resnet_model_compiles() {
    let p = PathBuf::from("../../spec/test-vectors/models/resnet9-fp32.rune");
    let m = rune_model::load(&p).expect("resnet fixture");
    let mut ir = rune_compiler::build_from_model(&m, "portable", "generic-x86-64").expect("ir");
    rune_compiler::select_kernels(&mut ir);
    let errs = rune_ir::verify(&ir);
    assert!(errs.is_empty(), "{:?}", errs);
    let kinds: Vec<&str> = ir.ops.iter().map(|o| o.kind.as_str()).collect();
    assert!(kinds.contains(&"StemConv"));
    assert!(kinds.contains(&"Policy"));
}

#[test]
fn classic_model_still_compiles() {
    let p = PathBuf::from("../../spec/test-vectors/models/tiny-mlp-fp32.rune");
    let m = rune_model::load(&p).expect("fixture");
    let mut ir = rune_compiler::build_from_model(&m, "portable", "generic-x86-64").expect("ir");
    rune_compiler::select_kernels(&mut ir);
    assert!(rune_ir::verify(&ir).is_empty());
}
