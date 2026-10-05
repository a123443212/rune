use rune_kernel as kernel;
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
    pub fn forward(&self, x: &[f32], out: &mut [f32], trace: Option<&mut MixerTrace>) {
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
        for a in 0..t {
            for b in 0..t {
                let val = s[a * t + b] + self.gab[a * t + b];
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
}
