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
use crate::resnet::ResnetWeights;

#[derive(Debug, Default)]
pub struct Scratch {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
    pub pooled: Vec<f32>,
    pub h: Vec<f32>,
    pub logits: Vec<f32>,
    pub probs: Vec<f32>,
}

impl Scratch {
    pub fn ensure(&mut self, board: usize, channels: usize, h2: usize, policy: usize) {
        let hw = board * board;
        let conv = channels * hw;
        if self.a.len() < conv {
            self.a.resize(conv, 0.0);
            self.b.resize(conv, 0.0);
            self.c.resize(conv, 0.0);
        }
        if self.pooled.len() < channels {
            self.pooled.resize(channels, 0.0);
        }
        if self.h.len() < h2 {
            self.h.resize(h2, 0.0);
        }
        if self.logits.len() < policy {
            self.logits.resize(policy, 0.0);
            self.probs.resize(policy, 0.0);
        }
    }
}

pub fn forward_fast(wt: &ResnetWeights, s: &mut Scratch, planes: &[f32]) -> (f32, [f32; 3], Vec<f32>) {
    let b = wt.cfg.board;
    let c = wt.cfg.channels;
    let hw = b * b;
    let h2 = wt.bh1.len();
    let ps = wt.policy.output;
    s.ensure(b, c, h2, ps);
    let a = &mut s.a[..c * hw];
    kernel::conv3x3_pad1_relu(planes, &wt.stem_w, Some(&wt.stem_b), a, wt.cfg.in_planes, c, b, b);
    {
        let n = c * hw;
        let (mut cur, mut nxt) = (&mut s.a[..n], &mut s.b[..n]);
        let tmp = &mut s.c[..n];
        for i in 0..wt.cfg.blocks {
            kernel::conv3x3_pad1_relu(cur, &wt.block_w1[i], Some(&wt.block_b1[i]), tmp, c, c, b, b);
            kernel::conv2d_nchw(tmp, &wt.block_w2[i], Some(&wt.block_b2[i]), nxt, 1, c, c, b, b, 3, 3, 1, 1);
            kernel::residual_add_relu_inplace(cur, nxt);
            std::mem::swap(&mut cur, &mut nxt);
        }
    }
    let fr = if wt.cfg.blocks % 2 == 0 { &s.a[..c * hw] } else { &s.b[..c * hw] };
    let pooled = &mut s.pooled[..c];
    kernel::global_avg_pool(fr, pooled, 1, c, b, b);
    let h = &mut s.h[..h2];
    kernel::mat_vec(&wt.vh1, pooled, Some(&wt.bh1), h, h2, c);
    for v in h.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        } else if *v > 1.0 {
            *v = 1.0;
        }
    }
    let mut vv = wt.bv;
    for i in 0..h2.min(wt.wv.len()) {
        vv += wt.wv[i] * h[i];
    }
    let value = vv.tanh();
    let mut wdl = [0.0f32; 3];
    kernel::mat_vec(&wt.wwdl, h, Some(&wt.bwdl), &mut wdl, 3, h2);
    let logits = &mut s.logits[..ps];
    kernel::mat_vec(&wt.policy.w, fr, Some(&wt.policy.b), logits, ps, c * hw);
    let probs = &mut s.probs[..ps];
    kernel::softmax(logits, probs);
    (value, wdl, probs.to_vec())
}
