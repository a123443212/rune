use crate::simd;

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
        let x = *v;
        *v = if x < 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { x };
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
