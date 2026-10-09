use rune_spec as spec;

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
