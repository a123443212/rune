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

pub struct Arena {
    buf: Vec<f32>,
}

impl Arena {
    pub fn new(bytes: usize) -> Self {
        let elems = bytes.div_ceil(4).max(1);
        Arena { buf: vec![0.0; elems] }
    }

    pub fn bytes(&self) -> usize {
        self.buf.len() * 4
    }

    pub fn slice(&self, offset_bytes: usize, elems: usize) -> &[f32] {
        let o = offset_bytes / 4;
        &self.buf[o..o + elems]
    }

    pub fn slice_mut(&mut self, offset_bytes: usize, elems: usize) -> &mut [f32] {
        let o = offset_bytes / 4;
        &mut self.buf[o..o + elems]
    }

    pub fn clear(&mut self) {
        for v in self.buf.iter_mut() {
            *v = 0.0;
        }
    }
}

pub fn arena_bytes_for(tokens: usize, dim: usize, h1: usize, h2: usize) -> usize {
    let live = [
        tokens * dim,
        tokens * dim,
        tokens * dim,
        tokens * dim,
        tokens * tokens,
        tokens * dim,
        h1,
        h2,
        tokens * dim,
    ];
    let mut total = 0;
    for e in live {
        total += e * 4 + 32;
    }
    total
}
