pub fn normalize_policy_inplace(probs: &mut [f32]) {
    let mut s = 0.0f32;
    for v in probs.iter() {
        if *v > 0.0 && v.is_finite() {
            s += *v;
        }
    }
    if s <= 0.0 || !s.is_finite() {
        let u = 1.0 / probs.len().max(1) as f32;
        for v in probs.iter_mut() {
            *v = u;
        }
        return;
    }
    for v in probs.iter_mut() {
        if *v <= 0.0 || !v.is_finite() {
            *v = 0.0;
        } else {
            *v /= s;
        }
    }
}

pub fn mask_policy(policy: &[f32], legal: &[bool], out: &mut [f32]) {
    for i in 0..policy.len().min(legal.len()).min(out.len()) {
        out[i] = if legal[i] { policy[i].max(0.0) } else { 0.0 };
    }
    normalize_policy_inplace(out);
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
