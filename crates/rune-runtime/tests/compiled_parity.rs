use std::fs;
use std::path::{Path, PathBuf};

use rune_runtime::board::Board;
use rune_runtime::compiled::CompiledEvaluator;
use rune_runtime::evaluator::Evaluator;
use rune_runtime::features::{compute_context, extract_features};

const EVAL_TOLERANCE: f32 = 2e-5;

fn assert_eval_close(
    label: &str,
    reference: &rune_runtime::evaluator::EvalResult,
    compiled: &rune_runtime::evaluator::EvalResult,
) {
    assert!(
        (reference.value - compiled.value).abs() <= EVAL_TOLERANCE,
        "{} value differs: {} vs {}",
        label,
        reference.value,
        compiled.value
    );
    for output in 0..3 {
        assert!(
            (reference.wdl[output] - compiled.wdl[output]).abs() <= EVAL_TOLERANCE,
            "{} WDL[{}] differs: {} vs {}",
            label,
            output,
            reference.wdl[output],
            compiled.wdl[output]
        );
    }
}

fn build_compiled_artifact(source: &Path, destination: &Path) {
    let model = rune_model::load(source).expect("load model fixture");
    let mut ir = rune_compiler::build_from_model(&model, "portable", "generic-x86-64")
        .expect("build portable IR");
    rune_compiler::fuse_plan(&mut ir);
    rune_compiler::select_kernels(&mut ir);
    let errors = rune_ir::verify(&ir);
    assert!(errors.is_empty(), "invalid IR: {:?}", errors);

    let kernel_plan_value = serde_json::to_value(&ir.kernel_plan).expect("kernel plan");
    let memory_plan_value = serde_json::to_value(&ir.memory).expect("memory plan");
    let target_value = serde_json::to_value(&ir.target).expect("target");
    let kernel_plan = serde_json::to_vec(&kernel_plan_value).expect("serialize kernel plan");
    let memory_plan = serde_json::to_vec(&memory_plan_value).expect("serialize memory plan");
    let target = serde_json::to_vec(&target_value).expect("serialize target");
    let plan_hash = rune_compiler::plan_hash(&kernel_plan, &memory_plan, &target);
    let payload = model.payload_bytes;
    let mut header = model.header.raw.clone();
    header["compiled"] = serde_json::Value::Bool(true);
    header["compiled_kind"] = serde_json::Value::String("compiled-v11".to_string());
    header["rune_ir_version"] = serde_json::Value::String(rune_ir::IR_VERSION.to_string());
    header["compiler_version"] = serde_json::Value::String(rune_ir::COMPILER_VERSION.to_string());
    header["target_isa"] = serde_json::Value::String("portable".to_string());
    header["target_cpu"] = serde_json::Value::String("generic-x86-64".to_string());
    header["kernel_plan"] = kernel_plan_value;
    header["fusion_plan"] = serde_json::to_value(&ir.fusion).expect("fusion plan");
    header["memory_plan"] = memory_plan_value;
    header["target"] = target_value;
    header["kernel_plan_hash"] = serde_json::Value::String(plan_hash);
    let source_header = serde_json::to_vec(&model.header.raw).expect("serialize source header");
    header["source_hash"] =
        serde_json::Value::String(rune_compiler::source_hash(&source_header, &payload));

    let bytes = rune_compiler::write_compiled(header, &payload);
    fs::write(destination, bytes).expect("write compiled fixture");
}

#[test]
fn reference_and_compiled_match_on_sparse_fixtures() {
    let positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ];
    let fixtures = [
        "tiny-mlp-fp32.rune",
        "small-gab-fp32.rune",
        "small-mh4-fp32.rune",
        "rel-08x32-fp32.rune",
    ];

    for (fixture_index, fixture) in fixtures.iter().enumerate() {
        let source = PathBuf::from(format!("../../spec/test-vectors/models/{}", fixture));
        let compiled = std::env::temp_dir().join(format!(
            "rune-compiled-parity-{}-{}.rune",
            std::process::id(),
            fixture_index
        ));
        build_compiled_artifact(&source, &compiled);

        let mut reference = Evaluator::load(&source).expect("load reference evaluator");
        let mut optimized = CompiledEvaluator::load(&compiled).expect("load compiled evaluator");
        for fen in positions {
            let board = Board::parse_fen(fen).expect("parse position");
            let reference_result = reference.evaluate_board(&board);
            let compiled_result = optimized.evaluate_board(&board);
            assert_eval_close(
                &format!("{} at {}", fixture, fen),
                &reference_result,
                &compiled_result,
            );
        }

        let before = Board::startpos();
        let mut after = before.clone();
        after.apply_uci("e2e4").expect("apply move");
        let before_features = extract_features(&before);
        let after_features = extract_features(&after);
        let context = compute_context(&after);
        reference.refresh(&before);
        optimized.refresh(&before);
        reference.update_incremental(&before_features, &after_features, &context);
        optimized.update_incremental(&before_features, &after_features, &context);
        let reference_incremental = reference.evaluate();
        let compiled_incremental = optimized.evaluate();
        assert_eval_close(fixture, &reference_incremental, &compiled_incremental);
        let reference_full = reference.evaluate_board(&after);
        let compiled_full = optimized.evaluate_board(&after);
        assert_eval_close(fixture, &reference_incremental, &reference_full);
        assert_eval_close(fixture, &compiled_incremental, &compiled_full);

        fs::remove_file(compiled).expect("remove compiled fixture");
    }
}
