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

use crate::error::{Result, RuntimeError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoBoard {
    pub size: usize,
    pub stones: Vec<i8>,
    pub stm: u8,
}

pub fn valid_size(n: usize) -> bool {
    n == 9 || n == 13 || n == 19
}

pub fn clamp01_f32(v: f32) -> f32 {
    if v < 0.0 {
        return 0.0;
    }
    if v > 1.0 {
        return 1.0;
    }
    v
}

pub fn neighbors_of(sq: usize, n: usize) -> Vec<usize> {
    let r = sq / n;
    let c = sq % n;
    let mut out = Vec::new();
    if r > 0 {
        out.push((r - 1) * n + c);
    }
    if r + 1 < n {
        out.push((r + 1) * n + c);
    }
    if c > 0 {
        out.push(r * n + (c - 1));
    }
    if c + 1 < n {
        out.push(r * n + (c + 1));
    }
    out
}

pub fn group_and_liberties(board: &[i8], start: usize, n: usize) -> (Vec<usize>, Vec<usize>) {
    let target = board[start];
    if target == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut seen = vec![false; n * n];
    let mut stones = Vec::new();
    let mut libs: Vec<usize> = Vec::new();
    let mut present = vec![false; n * n];
    let mut stack = vec![start];
    seen[start] = true;
    while let Some(cur) = stack.pop() {
        stones.push(cur);
        for nb in neighbors_of(cur, n) {
            let v = board[nb];
            if v == 0 {
                if !present[nb] {
                    present[nb] = true;
                    libs.push(nb);
                }
            } else if v == target && !seen[nb] {
                seen[nb] = true;
                stack.push(nb);
            }
        }
    }
    (stones, libs)
}

pub fn liberty_map(board: &[i8], n: usize) -> Vec<usize> {
    let mut out = vec![0usize; n * n];
    let mut done = vec![false; n * n];
    for sq in 0..n * n {
        if board[sq] == 0 || done[sq] {
            continue;
        }
        let (stones, libs) = group_and_liberties(board, sq, n);
        let k = libs.len();
        for s in stones {
            out[s] = k;
            done[s] = true;
        }
    }
    out
}

impl GoBoard {
    pub fn empty(size: usize) -> GoBoard {
        GoBoard { size, stones: vec![0; size * size], stm: 0 }
    }

    pub fn parse(s: &str) -> Result<GoBoard> {
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.is_empty() {
            return Err(RuntimeError::InvalidState("empty go state".to_string()));
        }
        let rows: Vec<&str> = parts[0].split('/').collect();
        let size = rows.len();
        if !valid_size(size) {
            return Err(RuntimeError::InvalidState("bad go size".to_string()));
        }
        let mut stones = vec![0i8; size * size];
        for (r, row) in rows.iter().enumerate() {
            if row.len() != size {
                return Err(RuntimeError::InvalidState("bad go row".to_string()));
            }
            for (c, ch) in row.chars().enumerate() {
                let v: i8 = match ch {
                    '.' => 0,
                    'X' => 1,
                    'O' => -1,
                    _ => return Err(RuntimeError::InvalidState("bad go stone".to_string())),
                };
                stones[r * size + c] = v;
            }
        }
        let stm: u8 = if parts.len() > 1 && (parts[1] == "w" || parts[1] == "O") { 1 } else { 0 };
        Ok(GoBoard { size, stones, stm })
    }

    pub fn at(&self, r: usize, c: usize) -> i8 {
        self.stones[r * self.size + c]
    }

    pub fn at_sq(&self, sq: usize) -> i8 {
        self.stones[sq]
    }

    pub fn liberties_of(&self, r: usize, c: usize) -> usize {
        let target = self.at(r, c);
        if target == 0 {
            return 0;
        }
        let (_, libs) = group_and_liberties(&self.stones, r * self.size + c, self.size);
        libs.len().min(8)
    }
}
