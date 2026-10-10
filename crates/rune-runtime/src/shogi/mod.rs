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

pub mod legal;
pub mod moves;
use crate::error::{Result, RuntimeError};
pub const SHOGI_GROUPS: usize = 9;
pub const SHOGI_TOKENS: usize = 8;
pub const SHOGI_DIM: usize = 32;
pub const SHOGI_CONTEXT_DIM: usize = 12;
pub const SHOGI_VOCABS: [usize; 9] = [648, 648, 972, 45, 45, 1134, 1134, 64, 1];
pub const SHOGI_FEATURE_VERSION: &str = "shogi_raw_v01";
pub const P: u8 = 0;
pub const L: u8 = 1;
pub const N: u8 = 2;
pub const S: u8 = 3;
pub const G: u8 = 4;
pub const B: u8 = 5;
pub const R: u8 = 6;
pub const K: u8 = 7;
pub const PP: u8 = 8;
pub const PL: u8 = 9;
pub const PN: u8 = 10;
pub const PS: u8 = 11;
pub const HB: u8 = 12;
pub const DR: u8 = 13;
pub const SBLACK: u8 = 0;
pub const SWHITE: u8 = 1;
const HAND_TYPES: usize = 7;
const HAND_MAX: [u8; 7] = [18, 4, 4, 4, 4, 2, 2];
const HAND_BASE: [u16; 7] = [0, 19, 24, 29, 34, 39, 42];
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SPiece {
    pub kind: u8,
    pub color: u8,
}
#[derive(Debug, Clone)]
pub struct ShogiBoard {
    pub sq: [Option<SPiece>; 81],
    pub stm: u8,
    pub hand: [[u8; 7]; 2],
    pub move_no: u32,
}
pub fn sq_file(sq: usize) -> i32 {
    (sq % 9) as i32
}
pub fn sq_rank(sq: usize) -> i32 {
    (sq / 9) as i32
}
pub fn make_sq(f: i32, r: i32) -> usize {
    (r * 9 + f) as usize
}
pub fn on_board(f: i32, r: i32) -> bool {
    f >= 0 && f < 9 && r >= 0 && r < 9
}
fn board_kind_index(c: char) -> Option<u8> {
    match c {
        'p' => Some(P),
        'l' => Some(L),
        'n' => Some(N),
        's' => Some(S),
        'g' => Some(G),
        'b' => Some(B),
        'r' => Some(R),
        'k' => Some(K),
        _ => None,
    }
}
fn hand_type_index(c: char) -> Option<usize> {
    match c {
        'p' => Some(0),
        'l' => Some(1),
        'n' => Some(2),
        's' => Some(3),
        'g' => Some(4),
        'b' => Some(5),
        'r' => Some(6),
        _ => None,
    }
}
fn parse_rank(rank: &str, row: usize, sq: &mut [Option<SPiece>; 81]) -> Result<()> {
    let mut f: i32 = 0;
    let mut chars = rank.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_ascii_digit() {
            f += c.to_digit(10).unwrap() as i32;
        } else {
            let mut promo = false;
            let mut pc = c;
            if pc == '+' {
                promo = true;
                pc = chars.next().ok_or_else(|| RuntimeError::BadFen("sfen: dangling +".to_string()))?;
            }
            let color = if pc.is_ascii_uppercase() { SBLACK } else { SWHITE };
            let base = board_kind_index(pc.to_ascii_lowercase()).ok_or_else(|| RuntimeError::BadFen("sfen: bad piece".to_string()))?;
            let kind = if promo {
                match base {
                    P => PP,
                    L => PL,
                    N => PN,
                    S => PS,
                    B => HB,
                    R => DR,
                    _ => return Err(RuntimeError::BadFen("sfen: bad promo".to_string())),
                }
            } else {
                base
            };
            if f >= 9 {
                return Err(RuntimeError::BadFen("sfen: rank overflow".to_string()));
            }
            sq[row * 9 + f as usize] = Some(SPiece { kind, color });
            f += 1;
        }
    }
    if f != 9 {
        return Err(RuntimeError::BadFen("sfen: rank width".to_string()));
    }
    Ok(())
}
impl ShogiBoard {
    pub fn parse_sfen(sfen: &str) -> Result<ShogiBoard> {
        let parts: Vec<&str> = sfen.split_whitespace().collect();
        if parts.is_empty() {
            return Err(RuntimeError::BadFen("sfen: empty".to_string()));
        }
        let ranks: Vec<&str> = parts[0].split('/').collect();
        if ranks.len() != 9 {
            return Err(RuntimeError::BadFen("sfen: need 9 ranks".to_string()));
        }
        let mut sq: [Option<SPiece>; 81] = [None; 81];
        for (ri, rank) in ranks.iter().enumerate() {
            parse_rank(rank, ri, &mut sq)?;
        }
        let stm = if parts.len() > 1 {
            match parts[1] {
                "b" => SBLACK,
                "w" => SWHITE,
                _ => return Err(RuntimeError::BadFen("sfen: bad side".to_string())),
            }
        } else {
            SBLACK
        };
        let mut hand = [[0u8; 7]; 2];
        if parts.len() > 2 && parts[2] != "-" {
            let mut num = String::new();
            for c in parts[2].chars() {
                if c.is_ascii_digit() {
                    num.push(c);
                } else {
                    let n: u32 = if num.is_empty() { 1 } else {
                        num.parse().map_err(|_| RuntimeError::BadFen("sfen: bad hand count".to_string()))?
                    };
                    num.clear();
                    let color = if c.is_ascii_uppercase() { 0 } else { 1 };
                    let ti = hand_type_index(c.to_ascii_lowercase())
                        .ok_or_else(|| RuntimeError::BadFen("sfen: bad hand piece".to_string()))?;
                    hand[color][ti] = hand[color][ti].saturating_add(n.min(255) as u8);
                }
            }
        }
        let move_no = if parts.len() > 3 {
            parts[3].parse().unwrap_or(1).max(1)
        } else {
            1
        };
        Ok(ShogiBoard { sq, stm, hand, move_no })
    }
    pub fn hand_total(&self) -> u32 {
        self.hand.iter().flatten().map(|v| *v as u32).sum()
    }
    pub fn promo_count(&self) -> u32 {
        self.sq.iter().filter(|c| matches!(c, Some(p) if p.kind >= PP)).count() as u32
    }
}
pub fn game_phase(hand_total: u32, promo_count: u32) -> u8 {
    let n = hand_total + promo_count;
    if n <= 3 {
        0
    } else if n <= 9 {
        1
    } else {
        2
    }
}
fn step_moves(kind: u8, color: u8) -> &'static [(i32, i32)] {
    match kind {
        P => {
            if color == SBLACK {
                &[(0, -1)]
            } else {
                &[(0, 1)]
            }
        }
        N => {
            if color == SBLACK {
                &[(-1, -2), (1, -2)]
            } else {
                &[(-1, 2), (1, 2)]
            }
        }
        S => {
            if color == SBLACK {
                &[(0, -1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
            } else {
                &[(0, 1), (-1, 1), (1, 1), (-1, -1), (1, -1)]
            }
        }
        G | PP | PL | PN | PS => {
            if color == SBLACK {
                &[(0, -1), (-1, -1), (1, -1), (-1, 0), (1, 0), (0, 1)]
            } else {
                &[(0, 1), (-1, 1), (1, 1), (-1, 0), (1, 0), (0, -1)]
            }
        }
        K => &[(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)],
        HB => &[(-1, 0), (1, 0), (0, -1), (0, 1)],
        DR => &[(-1, -1), (1, -1), (-1, 1), (1, 1)],
        _ => &[],
    }
}
fn slide_dirs(kind: u8) -> &'static [(i32, i32)] {
    match kind {
        B | HB => &[(-1, -1), (1, -1), (-1, 1), (1, 1)],
        R | DR => &[(-1, 0), (1, 0), (0, -1), (0, 1)],
        _ => &[],
    }
}
pub fn piece_attacks(b: &ShogiBoard, frm: usize, target: usize) -> bool {
    let (kind, color) = match b.sq[frm] {
        Some(p) => (p.kind, p.color),
        None => return false,
    };
    let (ff, rf) = (sq_file(frm), sq_rank(frm));
    let (tf, rt) = (sq_file(target), sq_rank(target));
    if kind == L {
        let f = if color == SBLACK { -1 } else { 1 };
        if tf != ff {
            return false;
        }
        let step = if rt > rf { 1 } else { -1 };
        if step != f {
            return false;
        }
        let mut r = rf + step;
        while r != rt {
            if b.sq[make_sq(tf, r)].is_some() {
                return false;
            }
            r += step;
        }
        return true;
    }
    for (df, dr) in step_moves(kind, color) {
        if ff + df == tf && rf + dr == rt {
            return true;
        }
    }
    for (df, dr) in slide_dirs(kind) {
        let (mut f, mut r) = (ff + df, rf + dr);
        while on_board(f, r) {
            if f == tf && r == rt {
                return true;
            }
            if b.sq[make_sq(f, r)].is_some() {
                break;
            }
            f += df;
            r += dr;
        }
    }
    false
}
fn minor_index(kind: u8) -> usize {
    match kind {
        P => 0,
        L => 1,
        N => 2,
        S => 3,
        G => 4,
        B => 5,
        R => 6,
        _ => 0,
    }
}
pub fn extract_features(b: &ShogiBoard) -> Vec<(u8, u16)> {
    let mut feats: Vec<(u8, u16)> = Vec::new();
    let us = b.stm;
    for sq in 0..81 {
        let cell = match b.sq[sq] {
            Some(c) => c,
            None => continue,
        };
        let ci = if cell.color == us { 0 } else { 1 } as u16;
        match cell.kind {
            P | L | N | S => feats.push((0, ci * 4 * 81 + cell.kind as u16 * 81 + sq as u16)),
            G | B | R | K => feats.push((1, ci * 4 * 81 + (cell.kind - 4) as u16 * 81 + sq as u16)),
            _ => feats.push((2, ci * 6 * 81 + (cell.kind - 8) as u16 * 81 + sq as u16)),
        }
        if cell.kind <= R && cell.kind != K {
            feats.push((6, ci * 7 * 81 + minor_index(cell.kind) as u16 * 81 + sq as u16));
        }
    }
    for ti in 0..HAND_TYPES {
        let cu = b.hand[us as usize][ti].min(HAND_MAX[ti]);
        let ct = b.hand[(1 - us) as usize][ti].min(HAND_MAX[ti]);
        feats.push((3, HAND_BASE[ti] + cu as u16));
        feats.push((4, HAND_BASE[ti] + ct as u16));
    }
    let mut kus = 81;
    for sq in 0..81 {
        match b.sq[sq] {
            Some(c) if c.kind == K && c.color == us => {
                kus = sq;
                break;
            }
            _ => {}
        }
    }
    let mut check = 0;
    if kus < 81 {
        for asq in 0..81 {
            match b.sq[asq] {
                Some(c) if c.color != us => {
                    if piece_attacks(b, asq, kus) {
                        check = 1;
                        feats.push((5, c.kind as u16 * 81 + asq as u16));
                    }
                }
                _ => {}
            }
        }
    }
    let hand_total = b.hand_total();
    let phase = game_phase(hand_total, b.promo_count());
    feats.push((7, us as u16));
    feats.push((7, (2 + (hand_total / 4).min(7)) as u16));
    feats.push((7, (10 + phase as u32) as u16));
    if check == 1 {
        feats.push((7, 13));
    }
    feats.push((7, (14 + (b.move_no / 20).min(5)) as u16));
    feats.sort_unstable();
    feats.dedup();
    feats
}
pub fn compute_context(b: &ShogiBoard) -> Vec<f32> {
    let us = b.stm;
    let mut us_hand = 0u32;
    let mut them_hand = 0u32;
    let mut us_promo = 0u32;
    let mut them_promo = 0u32;
    let mut us_board = 0u32;
    let mut them_board = 0u32;
    let mut kus = 81;
    for sq in 0..81 {
        let cell = match b.sq[sq] {
            Some(c) => c,
            None => continue,
        };
        let promo = cell.kind >= PP;
        if cell.color == us {
            us_board += 1;
            if cell.kind == K {
                kus = sq;
            }
            if promo {
                us_promo += 1;
            }
        } else {
            them_board += 1;
            if promo {
                them_promo += 1;
            }
        }
    }
    for c in 0..2 {
        for t in 0..HAND_TYPES {
            if c as u8 == us {
                us_hand += b.hand[c][t] as u32;
            } else {
                them_hand += b.hand[c][t] as u32;
            }
        }
    }
    let mut check = 0;
    if kus < 81 {
        for asq in 0..81 {
            match b.sq[asq] {
                Some(c) if c.color != us => {
                    if piece_attacks(b, asq, kus) {
                        check = 1;
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    let adv = if kus < 81 {
        let kr = sq_rank(kus) as f32;
        if us == SBLACK {
            (8.0 - kr) / 8.0
        } else {
            kr / 8.0
        }
    } else {
        0.0
    };
    let lead = (us_board + us_hand) as f32 - (them_board + them_hand) as f32;
    let phase = game_phase(us_hand + them_hand, us_promo + them_promo);
    let clip = |v: f32| {
        if v < 0.0 {
            0.0
        } else if v > 1.0 {
            1.0
        } else {
            v
        }
    };
    vec![
        clip(us as f32),
        clip(phase as f32 / 2.0),
        clip(us_hand as f32 / 20.0),
        clip(them_hand as f32 / 20.0),
        clip(us_promo as f32 / 8.0),
        clip(them_promo as f32 / 8.0),
        clip(us_board as f32 / 20.0),
        clip(them_board as f32 / 20.0),
        clip(check as f32),
        clip(adv),
        clip((lead + 20.0) / 40.0),
        clip(b.move_no as f32 / 200.0),
    ]
}
