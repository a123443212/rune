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

use super::board::liberty_map;
use super::state::{komi_bucket, move_bucket, GoState};

pub const GO_VOCABS: [usize; 9] = [361, 361, 361, 64, 64, 361, 361, 64, 18];
pub const GO_CONTEXT_DIM: usize = 12;
pub const FEATURE_VERSION_V02: &str = "go_planes_v02";
pub const PLANES_V02: usize = 8;

pub fn extract_planes_v02(st: &GoState) -> Vec<f32> {
    let n = st.board.size;
    let stm = st.board.stm;
    let libs = liberty_map(&st.board.stones, n);
    let mut out = vec![0.0f32; PLANES_V02 * n * n];
    let at = |p: usize, sq: usize| -> usize { p * n * n + sq };
    for r in 0..n {
        for c in 0..n {
            let sq = r * n + c;
            let v = st.board.stones[sq];
            let mine = (v == 1 && stm == 0) || (v == -1 && stm == 1);
            let theirs = (v == 1 || v == -1) && !mine && v != 0;
            if mine {
                out[at(0, sq)] = 1.0;
            }
            if theirs {
                out[at(1, sq)] = 1.0;
            }
            if v == 0 {
                out[at(2, sq)] = 1.0;
            } else {
                let k = libs[sq];
                if k <= 1 {
                    out[at(3, sq)] = 1.0;
                } else if k == 2 {
                    out[at(4, sq)] = 1.0;
                } else {
                    out[at(5, sq)] = 1.0;
                }
            }
            if let Some(k) = st.ko {
                if sq == k {
                    out[at(6, sq)] = 1.0;
                }
            }
            out[at(7, sq)] = stm as f32;
        }
    }
    out
}

pub fn token_features_v02(st: &GoState) -> Vec<(u8, u16)> {
    let n = st.board.size;
    let stm = st.board.stm;
    let libs = liberty_map(&st.board.stones, n);
    let mut feats: Vec<(u8, u16)> = Vec::new();
    let mut atari = 0usize;
    for sq in 0..n * n {
        let v = st.board.stones[sq];
        if v == 0 {
            continue;
        }
        let ci: usize = if (v == 1 && stm == 0) || (v == -1 && stm == 1) { 0 } else { 1 };
        let key = ((ci * 181 + sq) % 361) as u16;
        feats.push((0, key));
        feats.push((6, (sq % 361) as u16));
        let k = libs[sq];
        if k <= 2 {
            feats.push((1, key));
        }
        if k == 1 {
            feats.push((5, (sq % 361) as u16));
            atari += 1;
        }
    }
    if let Some(k) = st.ko {
        feats.push((2, (k % 361) as u16));
        feats.push((5, (360 - (k % 361)) as u16));
    }
    feats.push((7, stm as u16));
    feats.push((7, (2 + komi_bucket(st.ko.map(|_| st.komi).unwrap_or(st.komi))) as u16));
    feats.push((7, (10 + move_bucket(st.move_no)) as u16));
    let pn = if st.pass_no > 2 { 2 } else { st.pass_no } as u16;
    feats.push((7, 16 + pn));
    if st.ko.is_some() {
        feats.push((7, 19));
    }
    let mb = move_bucket(st.move_no);
    let ph: u16 = if mb <= 1 { 0 } else if mb <= 3 { 1 } else { 2 };
    feats.push((7, 20 + ph));
    let a = if atari > 8 { 8 } else { atari } as u16;
    feats.push((8, a));
    match st.ko {
        Some(k) => feats.push((8, (9 + (k % 9)) as u16)),
        None => feats.push((8, 9)),
    }
    feats.sort();
    feats.dedup();
    feats
}

pub fn phase_from_group7(groups: &[(u8, u16)]) -> u8 {
    let mut best: Option<u16> = None;
    for (g, i) in groups.iter() {
        if *g == 7 && *i >= 20 && *i <= 22 {
            match best {
                Some(b) => {
                    if *i > b {
                        best = Some(*i);
                    }
                }
                None => best = Some(*i),
            }
        }
    }
    match best {
        Some(v) => (v - 20) as u8,
        None => 0,
    }
}
