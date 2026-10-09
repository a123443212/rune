use rune_kernel as kernel;

pub const MH_HEADS: usize = 4;
pub const MH_HEAD_DIM: usize = 8;
pub const MH_TOKENS: usize = 8;
pub const MH_DIM: usize = 32;

#[derive(Debug, Clone)]
pub struct MultiHeadMixer {
    pub heads: usize,
    pub head_dim: usize,
    pub tokens: usize,
    pub dim: usize,
    pub wq: Vec<Vec<f32>>,
    pub bq: Vec<Vec<f32>>,
    pub wk: Vec<Vec<f32>>,
    pub bk: Vec<Vec<f32>>,
    pub wv: Vec<Vec<f32>>,
    pub bv: Vec<Vec<f32>>,
    pub gab: Vec<Vec<f32>>,
    pub wo: Vec<f32>,
    pub bwo: Vec<f32>,
    pub gate: kernel::Gate,
}

impl MultiHeadMixer {
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        let t = self.tokens;
        let d = self.dim;
        let hd = self.head_dim;
        let mut cat = vec![0.0_f32; t * d];
        for h in 0..self.heads {
            let mut q = vec![0.0_f32; t * hd];
            let mut k = vec![0.0_f32; t * hd];
            let mut vv = vec![0.0_f32; t * hd];
            for i in 0..t {
                let xb = &x[i * d..(i + 1) * d];
                let qb = &mut q[i * hd..(i + 1) * hd];
                let kb = &mut k[i * hd..(i + 1) * hd];
                let vb = &mut vv[i * hd..(i + 1) * hd];
                kernel::mat_vec(&self.wq[h], xb, Some(&self.bq[h]), qb, hd, d);
                kernel::mat_vec(&self.wk[h], xb, Some(&self.bk[h]), kb, hd, d);
                kernel::mat_vec(&self.wv[h], xb, Some(&self.bv[h]), vb, hd, d);
            }
            let mut s = vec![0.0_f32; t * t];
            kernel::mat_mul_tt(&q, &k, &mut s, t, t, hd);
            let mut g = vec![0.0_f32; t * t];
            for a in 0..t {
                for b in 0..t {
                    let val = s[a * t + b] + self.gab[h][a * t + b];
                    g[a * t + b] = self.gate.apply(val);
                }
            }
            let mut o = vec![0.0_f32; t * hd];
            kernel::mat_mul(&g, &vv, &mut o, t, hd, t);
            for i in 0..t {
                let dst = &mut cat[i * d + h * hd..i * d + (h + 1) * hd];
                let src = &o[i * hd..(i + 1) * hd];
                dst.copy_from_slice(src);
            }
        }
        let mut y = vec![0.0_f32; t * d];
        for i in 0..t {
            let xb = &cat[i * d..(i + 1) * d];
            let yb = &mut y[i * d..(i + 1) * d];
            kernel::mat_vec(&self.wo, xb, Some(&self.bwo), yb, d, d);
        }
        for i in 0..t * d {
            out[i] = x[i] + y[i];
        }
    }
}
