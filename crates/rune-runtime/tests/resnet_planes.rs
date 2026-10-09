use std::path::PathBuf;

#[test]
fn resnet_fixture_loads() {
    let p = PathBuf::from("../../spec/test-vectors/models/resnet9-fp32.rune");
    let m = rune_model::load(&p).expect("resnet fixture");
    assert_eq!(m.header.architecture_id, "RUNE-RESNET-01");
    assert_eq!(m.header.game, "go");
    let ev = rune_runtime::evaluator::Evaluator::from_model(&m).expect("resnet evaluator");
    assert!(ev.is_resnet());
}

#[test]
fn resnet_planes_deterministic() {
    let p = PathBuf::from("../../spec/test-vectors/models/resnet9-fp32.rune");
    let m = rune_model::load(&p).expect("fixture");
    let ev = rune_runtime::evaluator::Evaluator::from_model(&m).expect("evaluator");
    let go = rune_runtime::go::GoBoard::parse(&"........./........./........./........./........./........./........./........./......... b").expect("parse");
    let planes = rune_runtime::go::extract_planes(&go);
    let r1 = ev.evaluate_planes(&planes);
    let r2 = ev.evaluate_planes(&planes);
    assert!((r1.value - r2.value).abs() < 1e-9);
    assert_eq!(r1.policy.len(), r2.policy.len());
    assert!(!r1.policy.is_empty());
    let s: f32 = r1.policy.iter().sum();
    assert!((s - 1.0).abs() < 1e-5);
}

#[test]
fn resnet_matches_golden_value() {
    let raw = std::fs::read_to_string("../../spec/test-vectors/resnet/eval.json").expect("read golden");
    let g: serde_json::Value = serde_json::from_str(&raw).expect("parse");
    let p = PathBuf::from("../../spec/test-vectors/models/resnet9-fp32.rune");
    let m = rune_model::load(&p).expect("fixture");
    let ev = rune_runtime::evaluator::Evaluator::from_model(&m).expect("evaluator");
    let state = g["state"].as_str().unwrap();
    let go = rune_runtime::go::GoBoard::parse(state).expect("parse golden");
    let planes = rune_runtime::go::extract_planes(&go);
    let r = ev.evaluate_planes(&planes);
    let want = g["value"].as_f64().unwrap() as f32;
    assert!((r.value - want).abs() < 1e-4, "{} vs {}", r.value, want);
}
