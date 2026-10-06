use rune_kernel as kernel;
pub fn dyn_factors(dyn_u: &[f32], dyn_w: &[f32], tokens: usize, ctx_dim: usize, ctx: Option<&[f32]>) -> (Vec<f32>, Vec<f32>) {
    let mut du = vec![0.0_f32; tokens];
    let mut dw = vec![0.0_f32; tokens];
    let active = ctx.is_some() && ctx_dim > 0 && !dyn_u.is_empty() && !dyn_w.is_empty();
    if !active {
        return (du, dw);
    }
    let c = ctx.unwrap();
    for a in 0..tokens {
        let mut su = 0.0_f32;
        let mut sw = 0.0_f32;
        for i in 0..ctx_dim {
            su += dyn_u[a * ctx_dim + i] * c[i];
            sw += dyn_w[a * ctx_dim + i] * c[i];
        }
        du[a] = su;
        dw[a] = sw;
    }
    (du, dw)
}
#[derive(Debug, Clone)]
pub struct MixerWeights {
    pub tokens: usize,
    pub dim: usize,
    pub wq: Vec<f32>,
    pub bq: Vec<f32>,
    pub wk: Vec<f32>,
    pub bk: Vec<f32>,
    pub wv: Vec<f32>,
    pub bv: Vec<f32>,
    pub gab: Vec<f32>,
    pub dyn_u: Vec<f32>,
    pub dyn_w: Vec<f32>,
    pub ctx_dim: usize,
    pub gate_hard: bool,
    pub alpha: f32,
}
#[derive(Debug, Clone, Default)]
pub struct MixerTrace {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub scores: Vec<f32>,
    pub gates: Vec<f32>,
    pub mixed: Vec<f32>,
}
impl MixerWeights {
    pub fn dyn_active(&self, ctx: Option<&[f32]>) -> bool {
        ctx.is_some() && self.ctx_dim > 0 && !self.dyn_u.is_empty() && !self.dyn_w.is_empty()
    }
    pub fn forward(&self, x: &[f32], ctx: Option<&[f32]>, out: &mut [f32], trace: Option<&mut MixerTrace>) {
        let t = self.tokens;
        let d = self.dim;
        let mut q = vec![0.0_f32; t * d];
        let mut k = vec![0.0_f32; t * d];
        let mut vv = vec![0.0_f32; t * d];
        for i in 0..t {
            let xb = &x[i * d..(i + 1) * d];
            let qb = &mut q[i * d..(i + 1) * d];
            let kb = &mut k[i * d..(i + 1) * d];
            let vb = &mut vv[i * d..(i + 1) * d];
            kernel::mat_vec(&self.wq, xb, Some(&self.bq), qb, d, d);
            kernel::mat_vec(&self.wk, xb, Some(&self.bk), kb, d, d);
            kernel::mat_vec(&self.wv, xb, Some(&self.bv), vb, d, d);
        }
        let mut s = vec![0.0_f32; t * t];
        kernel::mat_mul_tt(&q, &k, &mut s, t, t, d);
        let mut g = vec![0.0_f32; t * t];
        let (du, dw) = dyn_factors(&self.dyn_u, &self.dyn_w, t, self.ctx_dim, ctx);
        let dyn_on = self.dyn_active(ctx);
        for a in 0..t {
            for b in 0..t {
                let mut val = s[a * t + b] + self.gab[a * t + b];
                if dyn_on {
                    val += kernel::clamp_delta(du[a] * dw[b]);
                }
                g[a * t + b] = if self.gate_hard { kernel::hard_sigmoid(val) } else { kernel::clipped_relu(val) };
            }
        }
        let mut y = vec![0.0_f32; t * d];
        kernel::mat_mul(&g, &vv, &mut y, t, d, t);
        for i in 0..t * d {
            out[i] = x[i] + self.alpha * y[i];
        }
        if let Some(tr) = trace {
            tr.q = q;
            tr.k = k;
            tr.v = vv;
            tr.scores = s;
            tr.gates = g;
            tr.mixed = out.to_vec();
        }
    }
}
#[derive(Debug, Clone)]
pub struct HeadWeights {
    pub input: usize,
    pub h1: usize,
    pub h2: usize,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub wvo: Vec<f32>,
    pub bvo: f32,
    pub wwdl: Vec<f32>,
    pub bwdl: Vec<f32>,
}
#[derive(Debug, Clone, Default)]
pub struct HeadTrace {
    pub h1: Vec<f32>,
    pub h2: Vec<f32>,
}
impl HeadWeights {
    pub fn forward(&self, flat: &[f32]) -> (f32, [f32; 3], HeadTrace) {
        let mut h1 = vec![0.0_f32; self.h1];
        let mut h2 = vec![0.0_f32; self.h2];
        kernel::mat_vec_clipped(&self.w1, flat, Some(&self.b1), &mut h1, self.h1, self.input);
        kernel::mat_vec_clipped(&self.w2, &h1, Some(&self.b2), &mut h2, self.h2, self.h1);
        let mut vv = self.bvo;
        for i in 0..self.h2 {
            vv += self.wvo[i] * h2[i];
        }
        let value = vv.tanh();
        let mut wdl = [0.0_f32; 3];
        kernel::mat_vec(&self.wwdl, &h2, Some(&self.bwdl), &mut wdl, 3, self.h2);
        (value, wdl, HeadTrace { h1, h2 })
    }
    pub fn forward_value_only(&self, flat: &[f32]) -> f32 {
        let mut h1 = vec![0.0_f32; self.h1];
        let mut h2 = vec![0.0_f32; self.h2];
        kernel::mat_vec_clipped(&self.w1, flat, Some(&self.b1), &mut h1, self.h1, self.input);
        kernel::mat_vec_clipped(&self.w2, &h1, Some(&self.b2), &mut h2, self.h2, self.h1);
        let mut vv = self.bvo;
        for i in 0..self.h2 {
            vv += self.wvo[i] * h2[i];
        }
        vv.tanh()
    }
}
