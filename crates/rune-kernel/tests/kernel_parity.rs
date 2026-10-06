use rune_kernel as kernel;
use std::fs;
fn vec_file(name: &str) -> serde_json::Value {
    let p = format!("../../spec/test-vectors/v10/{}", name);
    let s = fs::read_to_string(&p).expect("read vector");
    serde_json::from_str(&s).expect("parse vector")
}
#[test]
fn quant_matches_golden() {
    let v = vec_file("quant.json");
    let scale = v["scale"].as_f64().unwrap() as f32;
    let inputs = v["inputs"].as_array().unwrap();
    let e8 = v["int8"].as_array().unwrap();
    let e16 = v["int16"].as_array().unwrap();
    for i in 0..inputs.len() {
        let w = inputs[i].as_f64().unwrap() as f32;
        let q8 = kernel::quantize_half_away(w, scale, 127);
        let q16 = kernel::quantize_half_away(w, scale, 32767);
        assert_eq!(q8, e8[i].as_i64().unwrap() as i32);
        assert_eq!(q16, e16[i].as_i64().unwrap() as i32);
    }
    assert_eq!(kernel::quantize_half_away(f32::NAN, scale, 127), 0);
}
#[test]
fn matvec_matches_reference() {
    let v = vec_file("mixer.json");
    let dim = v["dim"].as_u64().unwrap() as usize;
    let wq: Vec<f32> = v["wq"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let bq: Vec<f32> = v["bq"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let input: Vec<f32> = v["input"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let exp_q: Vec<f32> = v["q"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let t = v["tokens"].as_u64().unwrap() as usize;
    let mut got = vec![0.0_f32; t * dim];
    kernel::clear_path_for_test();
    for i in 0..t {
        let xb = &input[i * dim..(i + 1) * dim];
        let qb = &mut got[i * dim..(i + 1) * dim];
        kernel::mat_vec_scalar(&wq, xb, Some(&bq), qb, dim, dim);
    }
    let (m, _) = kernel::max_abs_diff(&got, &exp_q);
    assert!(m < 1e-5, "max {}", m);
}
#[test]
fn clip_and_gate_exact() {
    assert_eq!(kernel::clipped_relu(-0.5), 0.0);
    assert_eq!(kernel::clipped_relu(0.5), 0.5);
    assert_eq!(kernel::clipped_relu(1.5), 1.0);
    assert!(kernel::clipped_relu(f32::NAN).is_nan());
    assert_eq!(kernel::clipped_relu(-0.0).to_bits(), (-0.0f32).to_bits());
    let h = kernel::hard_sigmoid(0.0);
    assert!((h - 0.5).abs() < 1e-6);
    assert_eq!(kernel::hard_sigmoid(-100.0), 0.0);
    assert_eq!(kernel::hard_sigmoid(100.0), 1.0);
    assert!(kernel::hard_sigmoid(f32::NAN).is_nan());
    assert!((kernel::hard_sigmoid(2.5) - 1.0).abs() < 1e-6);
}
#[test]
fn routing_nan_refines() {
    assert!(kernel::routing_refine(0.9, 0.5, 0.5, false, 0.5));
    assert!(!kernel::routing_refine(0.1, 0.5, 0.5, false, 0.5));
    assert!(kernel::routing_refine(f32::NAN, 0.5, 0.5, false, 0.5));
    assert!(!kernel::routing_refine(0.1, 0.5, 0.8, true, 0.2));
}
#[test]
fn fuzz_tokens_clip_bounded() {
    let mut s: u64 = 12345;
    let mut next = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64 / 4294967295.0) as f32 * 4.0 - 2.0
    };
    for _ in 0..200 {
        let x = next();
        let y = kernel::clipped_relu(x);
        assert!(y >= 0.0 && y <= 1.0);
    }
}
#[test]
fn fuzz_quant_saturates() {
    let mut s: u64 = 999;
    for _ in 0..500 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let w = ((s >> 33) as f64 / 4294967295.0) as f32 * 200.0 - 100.0;
        let q = kernel::quantize_half_away(w, 0.05, 127);
        assert!(q >= -127 && q <= 127);
        let q16 = kernel::quantize_half_away(w, 0.05, 32767);
        assert!(q16 >= -32767 && q16 <= 32767);
    }
}
