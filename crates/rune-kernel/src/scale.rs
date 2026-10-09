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
