use rune_kernel as kernel;
use rune_kernel::simd as ksimd;
use std::fs;
fn vec_file(name: &str) -> serde_json::Value {
    let p = format!("../../spec/test-vectors/v10/{}", name);
    let s = fs::read_to_string(&p).expect("read vector");
    serde_json::from_str(&s).expect("parse vector")
}
fn lcg_next(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 33) as f64 / 4294967295.0) as f32
}
const SHAPES: [(usize, usize); 8] = [(1, 1), (3, 32), (8, 8), (32, 128), (128, 256), (7, 24), (5, 40), (13, 17)];
fn worst_for_regime(mat_scale: f32, vec_lo: f32, vec_hi: f32, bias_scale: f32) -> f32 {
    let mut worst: f32 = 0.0;
    for (rows, cols) in SHAPES {
        let mut s: u64 = 20260701;
        let mat: Vec<f32> = (0..rows * cols).map(|_| (lcg_next(&mut s) * 2.0 - 1.0) * mat_scale).collect();
        let vec: Vec<f32> = (0..cols).map(|_| vec_lo + lcg_next(&mut s) * (vec_hi - vec_lo)).collect();
        let bias: Vec<f32> = (0..rows).map(|_| lcg_next(&mut s) * bias_scale).collect();
        let mut r = vec![0.0_f32; rows];
        let mut v = vec![0.0_f32; rows];
        kernel::mat_vec_scalar(&mat, &vec, Some(&bias), &mut r, rows, cols);
        ksimd::mat_vec_avx2(&mat, &vec, Some(&bias), &mut v, rows, cols);
        let (m, _) = kernel::max_abs_diff(&r, &v);
        if m > worst {
            worst = m;
        }
        let mut r0 = vec![0.0_f32; rows];
        let mut v0 = vec![0.0_f32; rows];
        kernel::mat_vec_scalar(&mat, &vec, None, &mut r0, rows, cols);
        ksimd::mat_vec_avx2(&mat, &vec, None, &mut v0, rows, cols);
        let (m0, _) = kernel::max_abs_diff(&r0, &v0);
        if m0 > worst {
            worst = m0;
        }
    }
    worst
}
#[test]
fn simd_lane_matches_scalar_within_tol() {
    if !ksimd::has_avx2_fma() {
        return;
    }
    kernel::set_path_for_test(kernel::KernelPath::Simd);
    let v = vec_file("mixer.json");
    let dim = v["dim"].as_u64().unwrap() as usize;
    let t = v["tokens"].as_u64().unwrap() as usize;
    let wq: Vec<f32> = v["wq"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let bq: Vec<f32> = v["bq"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let input: Vec<f32> = v["input"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let mut ref_out = vec![0.0_f32; t * dim];
    let mut simd_out = vec![0.0_f32; t * dim];
    for i in 0..t {
        let xb = &input[i * dim..(i + 1) * dim];
        kernel::mat_vec_scalar(&wq, xb, Some(&bq), &mut ref_out[i * dim..(i + 1) * dim], dim, dim);
        ksimd::mat_vec_avx2(&wq, xb, Some(&bq), &mut simd_out[i * dim..(i + 1) * dim], dim, dim);
    }
    let (m, _) = kernel::max_abs_diff(&ref_out, &simd_out);
    kernel::clear_path_for_test();
    assert!(m <= 1e-6, "simd vs scalar max {}", m);
}
#[test]
fn simd_net_magnitudes_within_spec_tol() {
    if !ksimd::has_avx2_fma() {
        return;
    }
    let w = worst_for_regime(0.08, 0.0, 1.0, 0.01);
    assert!(w <= 1e-5, "net regime worst {}", w);
}
#[test]
fn simd_stress_documents_fma_reassociation() {
    if !ksimd::has_avx2_fma() {
        return;
    }
    let w = worst_for_regime(1.0, -1.0, 1.0, 0.1);
    assert!(w <= 1e-4, "stress worst {}", w);
}
#[test]
fn dispatch_auto_matches_scalar() {
    kernel::clear_path_for_test();
    let mut s: u64 = 777;
    let (rows, cols) = (32, 128);
    let mat: Vec<f32> = (0..rows * cols).map(|_| lcg_next(&mut s)).collect();
    let vec: Vec<f32> = (0..cols).map(|_| lcg_next(&mut s)).collect();
    let mut r = vec![0.0_f32; rows];
    let mut d = vec![0.0_f32; rows];
    kernel::mat_vec_scalar(&mat, &vec, None, &mut r, rows, cols);
    kernel::mat_vec(&mat, &vec, None, &mut d, rows, cols);
    let (m, _) = kernel::max_abs_diff(&r, &d);
    assert!(m <= 1e-4, "dispatch max {}", m);
}
#[test]
fn fallback_never_crashes_without_avx2() {
    let (rows, cols) = (4, 32);
    let mat = vec![0.1_f32; rows * cols];
    let vec = vec![0.2_f32; cols];
    let bias = vec![0.01_f32; rows];
    let mut out = vec![0.0_f32; rows];
    let mut exp = vec![0.0_f32; rows];
    kernel::mat_vec_scalar(&mat, &vec, Some(&bias), &mut exp, rows, cols);
    ksimd::mat_vec_avx2(&mat, &vec, Some(&bias), &mut out, rows, cols);
    let (m, _) = kernel::max_abs_diff(&exp, &out);
    assert!(m <= 1e-6, "fallback max {}", m);
}
