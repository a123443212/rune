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

use crate::mixer::HeadTrace;

#[derive(Debug, Clone)]
pub struct SwiGluHead {
    pub input: usize,
    pub h1: usize,
    pub h2: usize,
    pub wgate: Vec<f32>,
    pub bgate: Vec<f32>,
    pub wup: Vec<f32>,
    pub bup: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub wvo: Vec<f32>,
    pub bvo: f32,
    pub wwdl: Vec<f32>,
    pub bwdl: Vec<f32>,
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

impl SwiGluHead {
    pub fn forward(&self, flat: &[f32]) -> (f32, [f32; 3], HeadTrace) {
        let mut gate = vec![0.0f32; self.h1];
        let mut up = vec![0.0f32; self.h1];
        kernel::mat_vec(&self.wgate, flat, Some(&self.bgate), &mut gate, self.h1, self.input);
        kernel::mat_vec(&self.wup, flat, Some(&self.bup), &mut up, self.h1, self.input);
        let mut h1 = vec![0.0f32; self.h1];
        for i in 0..self.h1 {
            h1[i] = silu(gate[i]) * up[i];
        }
        let mut h2 = vec![0.0f32; self.h2];
        kernel::mat_vec_clipped(&self.w2, &h1, Some(&self.b2), &mut h2, self.h2, self.h1);
        let mut vv = self.bvo;
        for i in 0..self.h2 {
            vv += self.wvo[i] * h2[i];
        }
        let value = vv.tanh();
        let mut wdl = [0.0f32; 3];
        kernel::mat_vec(&self.wwdl, &h2, Some(&self.bwdl), &mut wdl, 3, self.h2);
        (value, wdl, HeadTrace { h1, h2 })
    }

    pub fn forward_value_only(&self, flat: &[f32]) -> f32 {
        let (v, _, _) = self.forward(flat);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::{silu, SwiGluHead};

    fn head() -> SwiGluHead {
        SwiGluHead {
            input: 256,
            h1: 128,
            h2: 32,
            wgate: vec![0.01; 128 * 256],
            bgate: vec![0.0; 128],
            wup: vec![0.02; 128 * 256],
            bup: vec![0.0; 128],
            w2: vec![0.03; 32 * 128],
            b2: vec![0.0; 32],
            wvo: vec![0.04; 32],
            bvo: 0.0,
            wwdl: vec![0.05; 3 * 32],
            bwdl: vec![0.0; 3],
        }
    }

    #[test]
    fn silu_matches_definition() {
        assert!((silu(0.0) - 0.0).abs() < 1e-7);
        assert!((silu(1.0) - 0.7310586).abs() < 1e-6);
    }

    #[test]
    fn forward_shapes_and_bounds() {
        let h = head();
        let flat = vec![0.5f32; 256];
        let (v, wdl, tr) = h.forward(&flat);
        assert!(v >= -1.0 && v <= 1.0);
        assert_eq!(wdl.len(), 3);
        assert_eq!(tr.h1.len(), 128);
        assert_eq!(tr.h2.len(), 32);
        for x in wdl.iter() {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn zero_input_gives_zero_value() {
        let h = head();
        let flat = vec![0.0f32; 256];
        let (v, _, _) = h.forward(&flat);
        assert!(v.abs() < 1e-6);
    }
}
