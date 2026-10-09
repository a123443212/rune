use rune_runtime::mixer::{HeadWeights, MixerTrace, MixerWeights};
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
        gate: rune_kernel::Gate::Clip,
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
#[test]
fn head_pair_concat_matches_manual() {
    let h1 = 4usize;
    let h2 = 2usize;
    let input = 3usize;
    let flat = vec![0.5f32, -0.25, 1.0];
    let w1 = vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        0.5, 0.5, 0.5,
    ];
    let b1 = vec![0.0, 0.5, -0.5, 0.1];
    let w2 = vec![
        1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0,
    ];
    let b2 = vec![0.0, 0.0];
    let hw = HeadWeights {
        input, h1, h2,
        w1, b1, w2, b2,
        wvo: vec![1.0, -1.0], bvo: 0.0,
        wwdl: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], bwdl: vec![0.0, 0.0, 0.0],
    };
    assert!(hw.is_pair());
    let (value, wdl, tr) = hw.forward(&flat);
    let pre = [0.5f32, 0.25, 0.5, 0.6];
    let mut h1p = vec![0.0f32; 8];
    for i in 0..4 {
        h1p[i] = pre[i];
        h1p[4 + i] = pre[i] * pre[i];
    }
    assert!((tr.h1[0] - 0.5).abs() < 1e-6);
    assert!((tr.h1[4] - 0.25).abs() < 1e-6);
    assert!((tr.h2[0] - 0.625).abs() < 1e-6);
    assert!((tr.h2[1] - 0.28125).abs() < 1e-6);
    assert!((value - (0.625f32 - 0.28125).tanh()).abs() < 1e-6);
    assert!((wdl[0] - 0.625).abs() < 1e-6);
    let single = HeadWeights {
        input, h1, h2,
        w1: hw.w1.clone(), b1: hw.b1.clone(),
        w2: hw.w2[..h2 * h1].to_vec(), b2: hw.b2.clone(),
        wvo: hw.wvo.clone(), bvo: hw.bvo,
        wwdl: hw.wwdl.clone(), bwdl: hw.bwdl.clone(),
    };
    assert!(!single.is_pair());
    let (sv, _, _) = single.forward(&flat);
    assert!((sv - value).abs() > 1e-6);
}
