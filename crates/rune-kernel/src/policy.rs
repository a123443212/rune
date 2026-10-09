pub fn softmax(logits: &[f32], out: &mut [f32]) {
    debug_assert!(logits.len() == out.len());
    let mut m = f32::NEG_INFINITY;
    for v in logits.iter() {
        if *v > m {
            m = *v;
        }
    }
    let mut s = 0.0f32;
    for i in 0..logits.len() {
        let e = (logits[i] - m).exp();
        out[i] = e;
        s += e;
    }
    if s <= 0.0 || !s.is_finite() {
        let u = 1.0 / logits.len() as f32;
        for v in out.iter_mut() {
            *v = u;
        }
        return;
    }
    for v in out.iter_mut() {
        *v /= s;
    }
}

pub fn normalize_policy(logits: &[f32], legal: &[bool], out: &mut [f32]) {
    debug_assert!(logits.len() == legal.len());
    debug_assert!(logits.len() == out.len());
    let mut m = f32::NEG_INFINITY;
    for i in 0..logits.len() {
        if legal[i] && logits[i] > m {
            m = logits[i];
        }
    }
    if m == f32::NEG_INFINITY {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let mut s = 0.0f32;
    for i in 0..logits.len() {
        if legal[i] {
            let e = (logits[i] - m).exp();
            out[i] = e;
            s += e;
        } else {
            out[i] = 0.0;
        }
    }
    if s <= 0.0 || !s.is_finite() {
        let mut cnt = 0usize;
        for f in legal.iter() {
            if *f {
                cnt += 1;
            }
        }
        let u = if cnt == 0 { 0.0 } else { 1.0 / cnt as f32 };
        for i in 0..out.len() {
            out[i] = if legal[i] { u } else { 0.0 };
        }
        return;
    }
    for v in out.iter_mut() {
        *v /= s;
    }
}

pub fn policy_entropy(probs: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for p in probs.iter() {
        if *p > 0.0 {
            h -= *p * p.ln();
        }
    }
    h
}
