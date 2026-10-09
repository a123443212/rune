use rune_spec as spec;
pub mod arena;
pub mod fused;
pub mod quant;
pub mod simd;
pub mod specialized;
pub use simd::{active_path, active_path_name, clear_path_for_test, set_path_for_test, uses_simd, KernelPath};
pub fn clipped_relu(x: f32) -> f32 {
    if x < 0.0 {
        return 0.0;
    }
    if x > 1.0 {
        return 1.0;
    }
    x
}
pub fn hard_sigmoid(s: f32) -> f32 {
    let t = 0.2_f32 * s + 0.5_f32;
    if t < 0.0 {
        return 0.0;
    }
    if t > 1.0 {
        return 1.0;
    }
    t
}
pub fn screlu(s: f32) -> f32 {
    let c = clipped_relu(s);
    c * c
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gate {
    Clip,
    HardSigmoid,
    Screlu,
}
impl Gate {
    pub fn from_str(s: &str) -> Option<Gate> {
        match s {
            "clip" => Some(Gate::Clip),
            "hard_sigmoid" => Some(Gate::HardSigmoid),
            "screlu" => Some(Gate::Screlu),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Gate::Clip => "clip",
            Gate::HardSigmoid => "hard_sigmoid",
            Gate::Screlu => "screlu",
        }
    }
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Gate::Clip => clipped_relu(x),
            Gate::HardSigmoid => hard_sigmoid(x),
            Gate::Screlu => screlu(x),
        }
    }
}
pub fn clamp_delta(v: f32) -> f32 {
    if v < -0.25 {
        return -0.25;
    }
    if v > 0.25 {
        return 0.25;
    }
    v
}
pub fn quantize_half_away(w: f32, scale: f32, bound: i32) -> i32 {
    if !(scale > 0.0) {
        return 0;
    }
    if w.is_nan() {
        return 0;
    }
    if w > 1e30 {
        return bound;
    }
    if w < -1e30 {
        return -bound;
    }
    let q = w / scale;
    let r = if q >= 0.0 { (q + 0.5).floor() } else { (q - 0.5).ceil() };
    let mut v = r as i64;
    if v > bound as i64 {
        v = bound as i64;
    }
    if v < -(bound as i64) {
        v = -(bound as i64);
    }
    v as i32
}
pub fn dequantize(q: i32, scale: f32) -> f32 {
    q as f32 * scale
}
pub fn symmetric_scale(table: &[f32], bound: i32) -> f32 {
    let mut m: f32 = 0.0;
    for v in table {
        let a = v.abs();
        if a > m {
            m = a;
        }
    }
    if m <= 0.0 {
        return 1.0;
    }
    m / bound as f32
}
pub fn mat_vec_scalar(mat: &[f32], vec: &[f32], bias: Option<&[f32]>, out: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let mut acc: f32 = if let Some(b) = bias { b[r] } else { 0.0 };
        let base = r * cols;
        for c in 0..cols {
            acc += mat[base + c] * vec[c];
        }
        out[r] = acc;
    }
}
pub fn mat_vec(mat: &[f32], vec: &[f32], bias: Option<&[f32]>, out: &mut [f32], rows: usize, cols: usize) {
    simd::mat_vec_dispatched(mat, vec, bias, out, rows, cols);
}
pub fn mat_vec_clipped(mat: &[f32], vec: &[f32], bias: Option<&[f32]>, out: &mut [f32], rows: usize, cols: usize) {
    mat_vec(mat, vec, bias, out, rows, cols);
    for v in out.iter_mut().take(rows) {
        *v = clipped_relu(*v);
    }
}
pub fn mat_mul_tt(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc: f32 = 0.0;
            for t in 0..k {
                acc += a[i * k + t] * b[j * k + t];
            }
            out[i * n + j] = acc;
        }
    }
}
pub fn mat_mul(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc: f32 = 0.0;
            for t in 0..k {
                acc += a[i * k + t] * b[t * n + j];
            }
            out[i * n + j] = acc;
        }
    }
}
pub fn tokens_clip(acc: &[f32], tok: &mut [f32]) {
    for i in 0..acc.len() {
        tok[i] = clipped_relu(acc[i]);
    }
}
pub fn tokens_dequant_clip(acc: &[i32], scale: f32, tok: &mut [f32]) {
    for i in 0..acc.len() {
        tok[i] = clipped_relu(acc[i] as f32 * scale);
    }
}
pub fn routing_refine(score: f32, threshold: f32, t_high: f32, has_t_low: bool, t_low: f32) -> bool {
    if score.is_nan() {
        return true;
    }
    if score >= t_high {
        return true;
    }
    if has_t_low && score < t_low {
        return false;
    }
    score >= threshold
}
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut m: f32 = 0.0;
    let mut ai: usize = 0;
    for i in 0..a.len().min(b.len()) {
        let d = (a[i] - b[i]).abs();
        if d > m {
            m = d;
            ai = i;
        }
    }
    (m, ai)
}
pub fn spec_check() -> bool {
    let _ = spec::FEATURE_VERSION;
    true
}
