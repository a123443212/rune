use std::sync::atomic::{AtomicU8, Ordering};
use super::mat_vec_scalar;
static FORCED: AtomicU8 = AtomicU8::new(0);
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KernelPath {
    Auto,
    Scalar,
    Simd,
}
pub fn set_path_for_test(p: KernelPath) {
    let v = match p {
        KernelPath::Auto => 0,
        KernelPath::Scalar => 1,
        KernelPath::Simd => 2,
    };
    FORCED.store(v, Ordering::SeqCst);
}
pub fn clear_path_for_test() {
    FORCED.store(0, Ordering::SeqCst);
}
#[cfg(target_arch = "x86_64")]
pub fn has_avx2_fma() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}
#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx2_fma() -> bool {
    false
}
pub fn active_path() -> KernelPath {
    match FORCED.load(Ordering::SeqCst) {
        1 => KernelPath::Scalar,
        2 => KernelPath::Simd,
        _ => {
            if has_avx2_fma() {
                KernelPath::Simd
            } else {
                KernelPath::Scalar
            }
        }
    }
}
pub fn uses_simd() -> bool {
    active_path() == KernelPath::Simd
}
pub fn active_path_name() -> &'static str {
    if uses_simd() {
        "simd"
    } else {
        "scalar"
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mat_vec_avx2_unsafe(
    mat: *const f32,
    vec: *const f32,
    bias: *const f32,
    has_bias: bool,
    out: *mut f32,
    rows: usize,
    cols: usize,
) {
    use std::arch::x86_64::*;
    for r in 0..rows {
        let mut acc = _mm256_setzero_ps();
        let row = mat.add(r * cols);
        let mut c: usize = 0;
        while c + 8 <= cols {
            let a = _mm256_loadu_ps(row.add(c));
            let b = _mm256_loadu_ps(vec.add(c));
            acc = _mm256_fmadd_ps(a, b, acc);
            c += 8;
        }
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let mut s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        while c < cols {
            s += *row.add(c) * *vec.add(c);
            c += 1;
        }
        if has_bias {
            s += *bias.add(r);
        }
        *out.add(r) = s;
    }
}
pub fn mat_vec_avx2(
    mat: &[f32],
    vec: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    let need = rows.checked_mul(cols);
    let ok = match need {
        Some(n) => mat.len() >= n && vec.len() >= cols && out.len() >= rows,
        None => false,
    };
    let ok_bias = match bias {
        Some(b) => b.len() >= rows,
        None => true,
    };
    if !ok || !ok_bias || !has_avx2_fma() {
        mat_vec_scalar(mat, vec, bias, out, rows, cols);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let bp = match bias {
            Some(b) => b.as_ptr(),
            None => std::ptr::null(),
        };
        unsafe {
            mat_vec_avx2_unsafe(mat.as_ptr(), vec.as_ptr(), bp, bias.is_some(), out.as_mut_ptr(), rows, cols);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        mat_vec_scalar(mat, vec, bias, out, rows, cols);
    }
}
pub fn mat_vec_dispatched(
    mat: &[f32],
    vec: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    if active_path() == KernelPath::Simd {
        mat_vec_avx2(mat, vec, bias, out, rows, cols);
    } else {
        mat_vec_scalar(mat, vec, bias, out, rows, cols);
    }
}
