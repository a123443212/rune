use rune_kernel as kernel;

#[derive(Debug, Clone)]
pub struct IncrWeights {
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
    pub gate: kernel::Gate,
    pub alpha: f32,
}

#[derive(Debug, Clone, Default)]
pub struct CacheBytes {
    pub scores: usize,
    pub gates: usize,
    pub mixed: usize,
    pub scratch: usize,
}

#[derive(Debug, Clone, Default)]
struct Snapshot {
    x: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    s: Vec<f32>,
    g: Vec<f32>,
    y: Vec<f32>,
    out: Vec<f32>,
    du: Vec<f32>,
    dw: Vec<f32>,
    had_dyn: bool,
}

#[derive(Debug, Clone)]
pub struct RelationalCache {
    w: IncrWeights,
    threshold: usize,
    x: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    s: Vec<f32>,
    g: Vec<f32>,
    y: Vec<f32>,
    out: Vec<f32>,
    du: Vec<f32>,
    dw: Vec<f32>,
    had_dyn: bool,
    stack: Vec<Snapshot>,
    fallbacks: usize,
    incremental_updates: usize,
}

fn gate_fn(gate: kernel::Gate, x: f32) -> f32 {
    gate.apply(x)
}

impl RelationalCache {
    pub fn configure(w: IncrWeights, threshold: usize) -> RelationalCache {
        let td = w.tokens * w.dim;
        let tt = w.tokens * w.tokens;
        RelationalCache {
            du: vec![0.0; w.tokens],
            dw: vec![0.0; w.tokens],
            w,
            threshold,
            x: vec![0.0; td],
            q: vec![0.0; td],
            k: vec![0.0; td],
            v: vec![0.0; td],
            s: vec![0.0; tt],
            g: vec![0.0; tt],
            y: vec![0.0; td],
            out: vec![0.0; td],
            stack: Vec::new(),
            had_dyn: false,
            fallbacks: 0,
            incremental_updates: 0,
        }
    }

    pub fn use_incremental(&self, changed: &[usize]) -> bool {
        changed.len() <= self.threshold
    }

    pub fn threshold(&self) -> usize {
        self.threshold
    }

    pub fn set_threshold(&mut self, t: usize) {
        self.threshold = t;
    }

    pub fn fallbacks(&self) -> usize {
        self.fallbacks
    }

    pub fn incremental_updates(&self) -> usize {
        self.incremental_updates
    }

    pub fn tokens(&self) -> &[f32] {
        &self.x
    }

    pub fn out(&self) -> &[f32] {
        &self.out
    }

    pub fn scores(&self) -> &[f32] {
        &self.s
    }

    pub fn gates(&self) -> &[f32] {
        &self.g
    }

    fn dyn_active(&self, ctx: Option<&[f32]>) -> bool {
        ctx.is_some() && self.w.ctx_dim > 0 && !self.w.dyn_u.is_empty() && !self.w.dyn_w.is_empty()
    }

    fn dyn_factors(&self, ctx: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let t = self.w.tokens;
        let cd = self.w.ctx_dim;
        let mut du = vec![0.0; t];
        let mut dw = vec![0.0; t];
        for a in 0..t {
            let mut su = 0.0;
            let mut sw = 0.0;
            for c in 0..cd {
                su += self.w.dyn_u[a * cd + c] * ctx[c];
                sw += self.w.dyn_w[a * cd + c] * ctx[c];
            }
            du[a] = su;
            dw[a] = sw;
        }
        (du, dw)
    }

    fn full_into(&self, tokens: &[f32], ctx: Option<&[f32]>, q: &mut [f32], k: &mut [f32], vv: &mut [f32], s: &mut [f32], g: &mut [f32], y: &mut [f32], out: &mut [f32], du: &mut [f32], dw: &mut [f32]) {
        let t = self.w.tokens;
        let d = self.w.dim;
        for i in 0..t {
            let xb = &tokens[i * d..(i + 1) * d];
            let qb = &mut q[i * d..(i + 1) * d];
            let kb = &mut k[i * d..(i + 1) * d];
            let vb = &mut vv[i * d..(i + 1) * d];
            kernel::mat_vec(&self.w.wq, xb, Some(&self.w.bq), qb, d, d);
            kernel::mat_vec(&self.w.wk, xb, Some(&self.w.bk), kb, d, d);
            kernel::mat_vec(&self.w.wv, xb, Some(&self.w.bv), vb, d, d);
        }
        kernel::mat_mul_tt(q, k, s, t, t, d);
        let dyn_on = self.dyn_active(ctx);
        if dyn_on {
            let (fdu, fdw) = self.dyn_factors(ctx.unwrap());
            du.copy_from_slice(&fdu);
            dw.copy_from_slice(&fdw);
        }
        for a in 0..t {
            for b in 0..t {
                let mut val = s[a * t + b] + self.w.gab[a * t + b];
                if dyn_on {
                    val += kernel::clamp_delta(du[a] * dw[b]);
                }
                s[a * t + b] = val;
                g[a * t + b] = gate_fn(self.w.gate, val);
            }
        }
        kernel::mat_mul(g, vv, y, t, d, t);
        for i in 0..t * d {
            out[i] = tokens[i] + self.w.alpha * y[i];
        }
    }

    pub fn rebuild(&mut self, tokens: &[f32], ctx: Option<&[f32]>) {
        self.x.copy_from_slice(tokens);
        let mut q = self.q.clone();
        let mut k = self.k.clone();
        let mut vv = self.v.clone();
        let mut s = self.s.clone();
        let mut g = self.g.clone();
        let mut y = self.y.clone();
        let mut out = self.out.clone();
        let mut du = self.du.clone();
        let mut dw = self.dw.clone();
        self.full_into(tokens, ctx, &mut q, &mut k, &mut vv, &mut s, &mut g, &mut y, &mut out, &mut du, &mut dw);
        self.q = q;
        self.k = k;
        self.v = vv;
        self.s = s;
        self.g = g;
        self.y = y;
        self.out = out;
        self.du = du;
        self.dw = dw;
        self.had_dyn = self.dyn_active(ctx);
        self.stack.clear();
    }

    pub fn update(&mut self, tokens_new: &[f32], ctx: Option<&[f32]>, changed: &[usize]) {
        if !self.use_incremental(changed) {
            self.fallbacks += 1;
            self.rebuild(tokens_new, ctx);
            return;
        }
        self.incremental_updates += 1;
        let t = self.w.tokens;
        let d = self.w.dim;
        let mut mark = vec![false; t];
        for c in changed {
            if *c < t {
                mark[*c] = true;
            }
        }
        for c in changed {
            if *c >= t {
                continue;
            }
            let xb = &tokens_new[c * d..(c + 1) * d];
            let qb = &mut self.q[c * d..(c + 1) * d];
            let kb = &mut self.k[c * d..(c + 1) * d];
            let vb = &mut self.v[c * d..(c + 1) * d];
            kernel::mat_vec(&self.w.wq, xb, Some(&self.w.bq), qb, d, d);
            kernel::mat_vec(&self.w.wk, xb, Some(&self.w.bk), kb, d, d);
            kernel::mat_vec(&self.w.wv, xb, Some(&self.w.bv), vb, d, d);
        }
        let dyn_on = self.dyn_active(ctx);
        if dyn_on {
            let (fdu, fdw) = self.dyn_factors(ctx.unwrap());
            self.du.copy_from_slice(&fdu);
            self.dw.copy_from_slice(&fdw);
        }
        let (du, dw) = (self.du.clone(), self.dw.clone());
        let (gate, alpha) = (self.w.gate, self.w.alpha);
        let (q, k, s, g, gab) = (&self.q, &self.k, &mut self.s, &mut self.g, &self.w.gab);
        let gab_ref = gab;
        for a in 0..t {
            for b in 0..t {
                if !dyn_on && !self.had_dyn && !mark[a] && !mark[b] {
                    continue;
                }
                let mut dot = 0.0;
                for dd in 0..d {
                    dot += q[a * d + dd] * k[b * d + dd];
                }
                let mut val = dot + gab_ref[a * t + b];
                if dyn_on {
                    val += kernel::clamp_delta(du[a] * dw[b]);
                }
                s[a * t + b] = val;
                g[a * t + b] = gate_fn(gate, val);
            }
        }
        let vv = self.v.clone();
        kernel::mat_mul(&self.g, &vv, &mut self.y, t, d, t);
        self.x.copy_from_slice(tokens_new);
        for i in 0..t * d {
            self.out[i] = self.x[i] + alpha * self.y[i];
        }
        self.had_dyn = dyn_on;
    }

    pub fn push(&mut self) {
        self.stack.push(Snapshot {
            x: self.x.clone(),
            q: self.q.clone(),
            k: self.k.clone(),
            v: self.v.clone(),
            s: self.s.clone(),
            g: self.g.clone(),
            y: self.y.clone(),
            out: self.out.clone(),
            du: self.du.clone(),
            dw: self.dw.clone(),
            had_dyn: self.had_dyn,
        });
    }

    pub fn pop(&mut self) {
        let s = self.stack.pop().expect("pop without push");
        self.x = s.x;
        self.q = s.q;
        self.k = s.k;
        self.v = s.v;
        self.s = s.s;
        self.g = s.g;
        self.y = s.y;
        self.out = s.out;
        self.du = s.du;
        self.dw = s.dw;
        self.had_dyn = s.had_dyn;
    }

    pub fn verify_against_full(&self, tokens_new: &[f32], ctx: Option<&[f32]>, tol: f32) -> (bool, f32) {
        let t = self.w.tokens;
        let d = self.w.dim;
        let mut q = vec![0.0; t * d];
        let mut k = vec![0.0; t * d];
        let mut vv = vec![0.0; t * d];
        let mut s = vec![0.0; t * t];
        let mut g = vec![0.0; t * t];
        let mut y = vec![0.0; t * d];
        let mut out = vec![0.0; t * d];
        let mut du = vec![0.0; t];
        let mut dw = vec![0.0; t];
        self.full_into(tokens_new, ctx, &mut q, &mut k, &mut vv, &mut s, &mut g, &mut y, &mut out, &mut du, &mut dw);
        let pairs: [(&[f32], &[f32]); 7] = [
            (&self.q, &q),
            (&self.k, &k),
            (&self.v, &vv),
            (&self.s, &s),
            (&self.g, &g),
            (&self.y, &y),
            (&self.out, &out),
        ];
        let mut worst = 0.0;
        for (a, b) in pairs {
            for i in 0..a.len() {
                let dd = (a[i] - b[i]).abs();
                if dd > worst {
                    worst = dd;
                }
            }
        }
        (worst <= tol, worst)
    }

    pub fn cache_bytes(&self) -> CacheBytes {
        CacheBytes {
            scores: self.s.len() * 4,
            gates: self.g.len() * 4,
            mixed: self.y.len() * 4,
            scratch: 0,
        }
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use crate::interaction_graph::InteractionGraph;

    fn weights() -> IncrWeights {
        let d = 32;
        let mut wq = vec![0.0; d * d];
        let mut wk = vec![0.0; d * d];
        let mut wv = vec![0.0; d * d];
        let mut s: u64 = 777;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / 4294967295.0 - 0.5) as f32 * 0.16
        };
        for x in wq.iter_mut() { *x = rnd(); }
        for x in wk.iter_mut() { *x = rnd(); }
        for x in wv.iter_mut() { *x = rnd(); }
        IncrWeights {
            tokens: 8,
            dim: 32,
            wq,
            bq: vec![0.01; d],
            wk,
            bk: vec![0.01; d],
            wv,
            bv: vec![0.01; d],
            gab: vec![0.05; 64],
            dyn_u: Vec::new(),
            dyn_w: Vec::new(),
            ctx_dim: 0,
            gate: kernel::Gate::Clip,
            alpha: 1.0,
        }
    }

    #[test]
    fn parity_one_token() {
        let w = weights();
        let mut c = RelationalCache::configure(w, 8);
        let x0: Vec<f32> = (0..256).map(|i| (i * 37 % 100) as f32 / 100.0).collect();
        let mut x1 = x0.clone();
        for d in 0..32 { x1[3 * 32 + d] += 0.05; }
        c.rebuild(&x0, None);
        c.update(&x1, None, &[3]);
        let (ok, _) = c.verify_against_full(&x1, None, 1e-5);
        assert!(ok);
    }

    #[test]
    fn fallback_threshold() {
        let w = weights();
        let mut c = RelationalCache::configure(w, 1);
        let x0 = vec![0.5; 256];
        let mut x1 = vec![0.5; 256];
        x1[100] = 0.7;
        c.rebuild(&x0, None);
        c.update(&x1, None, &[3]);
        assert_eq!(c.incremental_updates(), 1);
        c.update(&x1, None, &[0, 5]);
        assert_eq!(c.fallbacks(), 1);
        let (ok, _) = c.verify_against_full(&x1, None, 1e-5);
        assert!(ok);
    }

    #[test]
    fn stack_isolation() {
        let w = weights();
        let mut c = RelationalCache::configure(w, 8);
        let x0 = vec![0.3; 256];
        let mut xa = vec![0.3; 256];
        let mut xb = vec![0.3; 256];
        xa[10] = 0.9;
        xb[200] = 0.1;
        c.rebuild(&x0, None);
        c.push();
        c.update(&xa, None, &[0]);
        let (ok_a, _) = c.verify_against_full(&xa, None, 1e-5);
        assert!(ok_a);
        c.pop();
        c.update(&xb, None, &[6]);
        let (ok_b, _) = c.verify_against_full(&xb, None, 1e-5);
        assert!(ok_b);
        assert!((c.tokens()[200] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn graph_dense_counts() {
        let g = InteractionGraph::dense(8);
        assert!(g.is_dense());
        assert_eq!(g.affected_edges(&[3]).len(), 15);
        assert_eq!(InteractionGraph::score_cells_for_changed(8, 2), 28);
    }

    #[test]
    fn shared_vectors_with_python_and_cpp() {
        let text = include_str!("../../../spec/test-vectors/v13/incremental.txt");
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        assert_eq!(lines.len(), 13);
        let h: Vec<&str> = lines[0].split_whitespace().collect();
        let t: usize = h[0].parse().unwrap();
        let d: usize = h[1].parse().unwrap();
        let thr: usize = h[2].parse().unwrap();
        let alpha: f32 = h[4].parse().unwrap();
        let nums = |i: usize| -> Vec<f32> {
            lines[i].split_whitespace().map(|x| x.parse().unwrap()).collect()
        };
        let (wq, bq, wk, bk, wv, bv, gab) =
            (nums(1), nums(2), nums(3), nums(4), nums(5), nums(6), nums(7));
        let (x0, x1, exp_full) = (nums(8), nums(9), nums(10));
        let changed: Vec<usize> = lines[12].split_whitespace().map(|x| x.parse().unwrap()).collect();
        let w = IncrWeights { tokens: t, dim: d, wq, bq, wk, bk, wv, bv, gab, dyn_u: Vec::new(), dyn_w: Vec::new(), ctx_dim: 0, gate: kernel::Gate::Clip, alpha };
        let mut c = RelationalCache::configure(w, thr);
        c.rebuild(&x0, None);
        c.update(&x1, None, &changed);
        assert_eq!(c.out().len(), exp_full.len());
        for (a, b) in c.out().iter().zip(exp_full.iter()) {
            assert!((a - b).abs() <= 1e-5, "{} vs {}", a, b);
        }
        let (ok, _) = c.verify_against_full(&x1, None, 1e-5);
        assert!(ok);
    }

    #[test]
    fn parity_dynamic_bias() {
        let mut w = weights();
        w.dyn_u = vec![0.02; 8 * 8];
        w.dyn_w = vec![0.02; 8 * 8];
        w.ctx_dim = 8;
        let mut c = RelationalCache::configure(w, 8);
        let x0 = vec![0.4; 256];
        let mut x1 = vec![0.4; 256];
        x1[35] = 0.9;
        let ctx = vec![0.5; 8];
        c.rebuild(&x0, Some(&ctx));
        c.update(&x1, Some(&ctx), &[1]);
        let (ok, worst) = c.verify_against_full(&x1, Some(&ctx), 1e-5);
        assert!(ok, "{}", worst);
    }

    #[test]
    fn ctx_change_recomputes_all_cells() {
        let mut w = weights();
        w.dyn_u = vec![0.02; 8 * 8];
        w.dyn_w = vec![0.03; 8 * 8];
        w.ctx_dim = 8;
        let mut c = RelationalCache::configure(w, 8);
        let x0 = vec![0.4; 256];
        let ctx0 = vec![0.5; 8];
        let ctx1 = vec![0.9; 8];
        c.rebuild(&x0, Some(&ctx0));
        c.update(&x0, Some(&ctx1), &[]);
        let (ok, worst) = c.verify_against_full(&x0, Some(&ctx1), 1e-5);
        assert!(ok, "{}", worst);
        c.update(&x0, None, &[]);
        let (ok, worst) = c.verify_against_full(&x0, None, 1e-5);
        assert!(ok, "{}", worst);
    }
}
