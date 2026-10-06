use rune_runtime::mixer::{MixerTrace, MixerWeights};
use std::fs;
fn vec_file(name: &str) -> serde_json::Value {
    let p = format!("../../spec/test-vectors/v10/{}", name);
    serde_json::from_str(&fs::read_to_string(&p).expect("read vector")).expect("parse")
}
fn f32s(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}
#[test]
fn mixer_composition_matches_golden_with_live_gates() {
    let v = vec_file("mixer.json");
    let t = v["tokens"].as_u64().unwrap() as usize;
    let d = v["dim"].as_u64().unwrap() as usize;
    let w = MixerWeights {
        tokens: t,
        dim: d,
        wq: f32s(&v["wq"]),
        bq: f32s(&v["bq"]),
        wk: f32s(&v["wk"]),
        bk: f32s(&v["bk"]),
        wv: f32s(&v["wvv"]),
        bv: f32s(&v["bvv"]),
        gab: f32s(&v["gab"]),
        dyn_u: Vec::new(),
        dyn_w: Vec::new(),
        ctx_dim: 0,
        gate_hard: false,
        alpha: 1.0,
    };
    let input = f32s(&v["input"]);
    let mut out = vec![0.0_f32; t * d];
    let mut tr = MixerTrace::default();
    w.forward(&input, None, &mut out, Some(&mut tr));
    let gates = f32s(&v["gates"]);
    assert!(gates.iter().any(|x| *x > 0.0));
    assert_eq!(tr.q.len(), t * d);
    assert!(max_abs(&tr.q, &f32s(&v["q"])) < 1e-5);
    assert!(max_abs(&tr.k, &f32s(&v["k"])) < 1e-5);
    assert!(max_abs(&tr.v, &f32s(&v["v"])) < 1e-5);
    assert!(max_abs(&tr.scores, &f32s(&v["scores"])) < 1e-5);
    assert!(max_abs(&tr.gates, &gates) < 1e-6);
    assert!(max_abs(&tr.mixed, &f32s(&v["mixed"])) < 1e-5);
    assert!(max_abs(&tr.mixed, &input) > 1e-6);
}
