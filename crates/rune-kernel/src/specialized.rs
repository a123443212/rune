use super::{clipped_relu, hard_sigmoid};

pub fn matvec_32(mat: &[f32], vec: &[f32], bias: &[f32], out: &mut [f32]) {
    debug_assert!(mat.len() >= 1024);
    debug_assert!(vec.len() >= 32);
    debug_assert!(bias.len() >= 32);
    debug_assert!(out.len() >= 32);
    for r in 0..32 {
        let base = r * 32;
        let mut s0 = bias[r];
        let mut c = 0;
        while c < 32 {
            s0 += mat[base + c] * vec[c];
            s0 += mat[base + c + 1] * vec[c + 1];
            s0 += mat[base + c + 2] * vec[c + 2];
            s0 += mat[base + c + 3] * vec[c + 3];
            c += 4;
        }
        out[r] = s0;
    }
}

pub fn matvec_8x32_batch(w: &[f32], b: &[f32], x: &[f32], out: &mut [f32]) {
    debug_assert!(x.len() >= 256);
    debug_assert!(out.len() >= 256);
    for t in 0..8 {
        let xb = &x[t * 32..t * 32 + 32];
        let ob = &mut out[t * 32..t * 32 + 32];
        matvec_32(w, xb, b, ob);
    }
}

pub fn matvec_128x256(mat: &[f32], vec: &[f32], bias: &[f32], out: &mut [f32]) {
    debug_assert!(mat.len() >= 128 * 256);
    debug_assert!(vec.len() >= 256);
    for r in 0..128 {
        let base = r * 256;
        let mut acc = bias[r];
        let mut c = 0;
        while c < 256 {
            acc += mat[base + c] * vec[c];
            acc += mat[base + c + 1] * vec[c + 1];
            acc += mat[base + c + 2] * vec[c + 2];
            acc += mat[base + c + 3] * vec[c + 3];
            c += 4;
        }
        out[r] = acc;
    }
}

pub fn matvec_32x128_clipped(mat: &[f32], vec: &[f32], bias: &[f32], out: &mut [f32]) {
    debug_assert!(mat.len() >= 32 * 128);
    for r in 0..32 {
        let base = r * 128;
        let mut acc = bias[r];
        let mut c = 0;
        while c < 128 {
            acc += mat[base + c] * vec[c];
            acc += mat[base + c + 1] * vec[c + 1];
            acc += mat[base + c + 2] * vec[c + 2];
            acc += mat[base + c + 3] * vec[c + 3];
            c += 4;
        }
        out[r] = clipped_relu(acc);
    }
}

pub fn matvec_128x256_clipped(mat: &[f32], vec: &[f32], bias: &[f32], out: &mut [f32]) {
    debug_assert!(mat.len() >= 128 * 256);
    for r in 0..128 {
        let base = r * 256;
        let mut acc = bias[r];
        let mut c = 0;
        while c < 256 {
            acc += mat[base + c] * vec[c];
            acc += mat[base + c + 1] * vec[c + 1];
            acc += mat[base + c + 2] * vec[c + 2];
            acc += mat[base + c + 3] * vec[c + 3];
            c += 4;
        }
        out[r] = clipped_relu(acc);
    }
}

pub fn score_8x8_dim32(q: &[f32], k: &[f32], out: &mut [f32]) {
    debug_assert!(q.len() >= 256);
    debug_assert!(k.len() >= 256);
    debug_assert!(out.len() >= 64);
    for a in 0..8 {
        for b in 0..8 {
            let mut acc = 0.0;
            let qa = &q[a * 32..a * 32 + 32];
            let kb = &k[b * 32..b * 32 + 32];
            let mut t = 0;
            while t < 32 {
                acc += qa[t] * kb[t];
                acc += qa[t + 1] * kb[t + 1];
                acc += qa[t + 2] * kb[t + 2];
                acc += qa[t + 3] * kb[t + 3];
                t += 4;
            }
            out[a * 8 + b] = acc;
        }
    }
}

pub fn mix_8x8_32(g: &[f32], v: &[f32], out: &mut [f32]) {
    debug_assert!(g.len() >= 64);
    debug_assert!(v.len() >= 256);
    debug_assert!(out.len() >= 256);
    for a in 0..8 {
        for d in 0..32 {
            let mut acc = 0.0;
            for b in 0..8 {
                acc += g[a * 8 + b] * v[b * 32 + d];
            }
            out[a * 32 + d] = acc;
        }
    }
}

pub fn residual_8x32(x: &[f32], y: &[f32], alpha: f32, out: &mut [f32]) {
    debug_assert!(x.len() >= 256);
    debug_assert!(y.len() >= 256);
    for i in 0..256 {
        out[i] = x[i] + alpha * y[i];
    }
}

pub fn gate_clip_inplace(s: &mut [f32]) {
    for v in s.iter_mut() {
        *v = clipped_relu(*v);
    }
}

pub fn gate_hard_inplace(s: &mut [f32]) {
    for i in 0..s.len() {
        s[i] = hard_sigmoid(s[i]);
    }
}
