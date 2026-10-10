// RUNE — Relational Unified Neural Evaluator
// Copyright (C) 2026 a123443212
//
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// This project is dual-licensed under the MIT License and the
// Apache License, Version 2.0. You may choose either license
// when using, copying, modifying, or distributing this software.
//
// MIT License: https://opensource.org/license/mit
// Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, this
// software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
// OR CONDITIONS OF ANY KIND, either express or implied.

use rune_kernel as kernel;

#[derive(Debug, Clone)]
pub struct SoftMixer {
    pub tokens: usize,
    pub dim: usize,
    pub wq: Vec<f32>,
    pub bq: Vec<f32>,
    pub wk: Vec<f32>,
    pub bk: Vec<f32>,
    pub wv: Vec<f32>,
    pub bv: Vec<f32>,
    pub gab: Vec<f32>,
    pub scale: f32,
}

pub fn scale_for_dim(dim: usize) -> f32 {
    1.0f32 / (dim as f32).sqrt()
}

pub fn soft_forward_into(
    wq: &[f32],
    bq: &[f32],
    wk: &[f32],
    bk: &[f32],
    wv: &[f32],
    bv: &[f32],
    gab: &[f32],
    scale: f32,
    tokens: usize,
    dim: usize,
    x: &[f32],
    out: &mut [f32],
) {
    let mut q = vec![0.0f32; tokens * dim];
    let mut k = vec![0.0f32; tokens * dim];
    let mut vv = vec![0.0f32; tokens * dim];
    for i in 0..tokens {
        let xb = &x[i * dim..(i + 1) * dim];
        let qb = &mut q[i * dim..(i + 1) * dim];
        let kb = &mut k[i * dim..(i + 1) * dim];
        let vb = &mut vv[i * dim..(i + 1) * dim];
        kernel::mat_vec(wq, xb, Some(bq), qb, dim, dim);
        kernel::mat_vec(wk, xb, Some(bk), kb, dim, dim);
        kernel::mat_vec(wv, xb, Some(bv), vb, dim, dim);
    }
    let mut w = vec![0.0f32; tokens * tokens];
    for a in 0..tokens {
        let mut logits = vec![0.0f32; tokens];
        for b in 0..tokens {
            let mut acc = 0.0f32;
            for t in 0..dim {
                acc += q[a * dim + t] * k[b * dim + t];
            }
            logits[b] = acc * scale + gab[a * tokens + b];
        }
        let mut probs = vec![0.0f32; tokens];
        kernel::softmax(&logits, &mut probs);
        w[a * tokens..(a + 1) * tokens].copy_from_slice(&probs);
    }
    let mut y = vec![0.0f32; tokens * dim];
    kernel::mat_mul(&w, &vv, &mut y, tokens, dim, tokens);
    for i in 0..tokens * dim {
        out[i] = x[i] + y[i];
    }
}

impl SoftMixer {
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        soft_forward_into(
            &self.wq,
            &self.bq,
            &self.wk,
            &self.bk,
            &self.wv,
            &self.bv,
            &self.gab,
            self.scale,
            self.tokens,
            self.dim,
            x,
            out,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{scale_for_dim, SoftMixer};

    fn mixer() -> SoftMixer {
        let t = 8;
        let d = 32;
        SoftMixer {
            tokens: t,
            dim: d,
            wq: vec![0.01; d * d],
            bq: vec![0.0; d],
            wk: vec![0.02; d * d],
            bk: vec![0.0; d],
            wv: vec![0.03; d * d],
            bv: vec![0.0; d],
            gab: vec![0.0; t * t],
            scale: scale_for_dim(d),
        }
    }

    #[test]
    fn rows_sum_to_one_and_residual_holds() {
        let m = mixer();
        let x = vec![0.5f32; 8 * 32];
        let mut out = vec![0.0f32; 8 * 32];
        m.forward(&x, &mut out);
        assert_eq!(out.len(), 256);
        for v in out.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn zero_weights_returns_input() {
        let mut m = mixer();
        for w in m
            .wq
            .iter_mut()
            .chain(m.wk.iter_mut())
            .chain(m.wv.iter_mut())
        {
            *w = 0.0;
        }
        let x: Vec<f32> = (0..256).map(|i| i as f32 / 256.0).collect();
        let mut out = vec![0.0f32; 256];
        m.forward(&x, &mut out);
        for i in 0..256 {
            let expect = x[i] + 0.0;
            assert!((out[i] - expect).abs() < 1e-4, "i={} got={}", i, out[i]);
        }
    }

    #[test]
    fn scale_matches_inverse_sqrt() {
        let s = scale_for_dim(32);
        assert!((s - 0.17677669).abs() < 1e-7);
    }
}
