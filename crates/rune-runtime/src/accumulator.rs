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
pub struct Tables {
    pub dim: usize,
    pub data: [Vec<f32>; 9],
}
impl Tables {
    pub fn zeros(dim: usize, vocabs: [usize; 9]) -> Tables {
        Tables {
            dim,
            data: [
                vec![0.0; vocabs[0] * dim],
                vec![0.0; vocabs[1] * dim],
                vec![0.0; vocabs[2] * dim],
                vec![0.0; vocabs[3] * dim],
                vec![0.0; vocabs[4] * dim],
                vec![0.0; vocabs[5] * dim],
                vec![0.0; vocabs[6] * dim],
                vec![0.0; vocabs[7] * dim],
                vec![0.0; vocabs[8] * dim],
            ],
        }
    }
    pub fn row(&self, g: usize, idx: usize) -> &[f32] {
        let base = idx * self.dim;
        &self.data[g][base..base + self.dim]
    }
}
pub fn token_of(tokens: usize, group: u8, index: u16) -> Option<usize> {
    if group == 8 {
        return Some(0);
    }
    if tokens == 8 {
        return Some(group as usize);
    }
    if tokens == 6 {
        if group <= 2 {
            return Some(group as usize);
        }
        if group == 3 || group == 4 {
            return Some(3);
        }
        if group == 5 || group == 6 {
            return Some(4);
        }
        if group == 7 {
            return Some(5);
        }
        return None;
    }
    if tokens == 10 {
        if group <= 1 {
            return Some(group as usize);
        }
        if group == 2 {
            if index < 128 {
                return Some(2);
            }
            return Some(3);
        }
        if group == 3 {
            return Some(4);
        }
        if group == 4 {
            return Some(5);
        }
        if group == 5 {
            if index < 384 {
                return Some(6);
            }
            return Some(7);
        }
        if group == 6 {
            return Some(8);
        }
        if group == 7 {
            return Some(9);
        }
        return None;
    }
    None
}
pub fn token_scale_group(tokens: usize, token: usize) -> usize {
    if tokens == 8 {
        return token.min(7);
    }
    if tokens == 6 {
        const PRIMARY: [usize; 6] = [0, 1, 2, 3, 5, 7];
        return PRIMARY[token.min(5)];
    }
    if tokens == 10 {
        const PRIMARY: [usize; 10] = [0, 1, 2, 2, 3, 4, 5, 5, 6, 7];
        return PRIMARY[token.min(9)];
    }
    token.min(7)
}
pub struct Accumulator {
    pub dim: usize,
    pub tokens: usize,
    acc: Vec<f32>,
    stack: Vec<Vec<f32>>,
}
impl Accumulator {
    pub fn new(tokens: usize, dim: usize) -> Accumulator {
        Accumulator { tokens, dim, acc: vec![0.0; tokens * dim], stack: Vec::new() }
    }
    pub fn refresh(&mut self, tables: &Tables, feats: &[(u8, u16)]) {
        for v in self.acc.iter_mut() {
            *v = 0.0;
        }
        let (added, empty): (Vec<(u8, u16)>, Vec<(u8, u16)>) = (feats.to_vec(), Vec::new());
        self.apply_diff(tables, &added, &empty);
    }
    pub fn apply_diff(&mut self, tables: &Tables, added: &[(u8, u16)], removed: &[(u8, u16)]) {
        for (g, idx) in added {
            let t = match token_of(self.tokens, *g, *idx) {
                Some(v) => v,
                None => continue,
            };
            let row = tables.row(*g as usize, *idx as usize);
            let base = t * self.dim;
            for d in 0..self.dim {
                self.acc[base + d] += row[d];
            }
        }
        for (g, idx) in removed {
            let t = match token_of(self.tokens, *g, *idx) {
                Some(v) => v,
                None => continue,
            };
            let row = tables.row(*g as usize, *idx as usize);
            let base = t * self.dim;
            for d in 0..self.dim {
                self.acc[base + d] -= row[d];
            }
        }
    }
    pub fn tokens(&self, out: &mut [f32]) {
        kernel::tokens_clip(&self.acc, out);
    }
    pub fn push(&mut self) {
        self.stack.push(self.acc.clone());
    }
    pub fn pop(&mut self) -> bool {
        let prev = match self.stack.pop() {
            Some(v) => v,
            None => return false,
        };
        self.acc = prev;
        true
    }
    pub fn depth(&self) -> usize {
        self.stack.len()
    }
    pub fn snapshot(&self) -> Vec<f32> {
        self.acc.clone()
    }
    pub fn restore(&mut self, snap: &[f32]) {
        self.acc.copy_from_slice(snap);
    }
    pub fn raw(&self) -> &[f32] {
        &self.acc
    }
}
pub struct IntAccumulator {
    pub dim: usize,
    pub tokens: usize,
    acc: Vec<i32>,
    scales: [f32; 9],
}
impl IntAccumulator {
    pub fn new(tokens: usize, dim: usize, scales: [f32; 9]) -> IntAccumulator {
        IntAccumulator { tokens, dim, acc: vec![0; 9 * dim], scales }
    }
    pub fn refresh(&mut self, qtables: &[Vec<i32>; 9], feats: &[(u8, u16)]) {
        for v in self.acc.iter_mut() {
            *v = 0;
        }
        self.apply_diff(qtables, feats, &[]);
    }
    pub fn apply_diff(&mut self, qtables: &[Vec<i32>; 9], added: &[(u8, u16)], removed: &[(u8, u16)]) {
        for (g, idx) in added {
            if *g as usize >= 9 {
                continue;
            }
            let base_q = *idx as usize * self.dim;
            let base_a = *g as usize * self.dim;
            for d in 0..self.dim {
                self.acc[base_a + d] += qtables[*g as usize][base_q + d];
            }
        }
        for (g, idx) in removed {
            if *g as usize >= 9 {
                continue;
            }
            let base_q = *idx as usize * self.dim;
            let base_a = *g as usize * self.dim;
            for d in 0..self.dim {
                self.acc[base_a + d] -= qtables[*g as usize][base_q + d];
            }
        }
    }
    pub fn tokens(&self, out: &mut [f32]) {
        for t in 0..self.tokens {
            for d in 0..self.dim {
                out[t * self.dim + d] = 0.0;
            }
        }
        for g in 0..9 {
            let t = match token_of(self.tokens, g as u8, 0) {
                Some(v) => v,
                None => continue,
            };
            if t >= self.tokens {
                continue;
            }
            for d in 0..self.dim {
                out[t * self.dim + d] += self.acc[g * self.dim + d] as f32 * self.scales[g];
            }
        }
        for v in out.iter_mut() {
            *v = kernel::clipped_relu(*v);
        }
    }
}
