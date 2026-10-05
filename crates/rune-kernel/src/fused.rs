use super::specialized;

pub fn qkv_fused_8x32(wq: &[f32], bq: &[f32], wk: &[f32], bk: &[f32], wv: &[f32], bv: &[f32], x: &[f32], q: &mut [f32], k: &mut [f32], v: &mut [f32]) {
    for t in 0..8 {
        let xb = &x[t * 32..t * 32 + 32];
        let qb = &mut q[t * 32..t * 32 + 32];
        let kb = &mut k[t * 32..t * 32 + 32];
        let vb = &mut v[t * 32..t * 32 + 32];
        specialized::matvec_32(wq, xb, bq, qb);
        specialized::matvec_32(wk, xb, bk, kb);
        specialized::matvec_32(wv, xb, bv, vb);
    }
}

pub fn score_bias_gate_8x8(q: &[f32], k: &[f32], gab: &[f32], gate_hard: bool, scores: &mut [f32], gate: &mut [f32]) {
    specialized::score_8x8_dim32(q, k, scores);
    for i in 0..64 {
        let b = scores[i] + gab[i];
        scores[i] = b;
        gate[i] = if gate_hard {
            super::hard_sigmoid(b)
        } else {
            super::clipped_relu(b)
        };
    }
}

pub fn score_bias_gate_inplace(q: &[f32], k: &[f32], gab: &[f32], gate_hard: bool, buf: &mut [f32]) {
    specialized::score_8x8_dim32(q, k, buf);
    for i in 0..64 {
        let b = buf[i] + gab[i];
        buf[i] = if gate_hard {
            super::hard_sigmoid(b)
        } else {
            super::clipped_relu(b)
        };
    }
}

pub fn mix_residual_8x32(g: &[f32], v: &[f32], x: &[f32], alpha: f32, tmp: &mut [f32], out: &mut [f32]) {
    specialized::mix_8x8_32(g, v, tmp);
    specialized::residual_8x32(x, tmp, alpha, out);
}

pub fn linear_bias_clip_128x256(w1: &[f32], b1: &[f32], flat: &[f32], h1: &mut [f32]) {
    specialized::matvec_128x256_clipped(w1, flat, b1, h1);
}

pub fn linear_bias_clip_32x128(w2: &[f32], b2: &[f32], h1: &[f32], h2: &mut [f32]) {
    specialized::matvec_32x128_clipped(w2, h1, b2, h2);
}

pub fn dot_tanh(wvo: &[f32], bvo: f32, h2: &[f32]) -> f32 {
    let mut acc = bvo;
    for i in 0..h2.len() {
        acc += wvo[i] * h2[i];
    }
    acc.tanh()
}

pub fn wdl_3x32(wwdl: &[f32], bwdl: &[f32], h2: &[f32], wdl: &mut [f32; 3]) {
    for r in 0..3 {
        let mut acc = bwdl[r];
        for c in 0..32 {
            acc += wwdl[r * 32 + c] * h2[c];
        }
        wdl[r] = acc;
    }
}
