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
pub const XIANGQI_GROUPS: usize = 9;
pub const XIANGQI_TOKENS: usize = 8;
pub const XIANGQI_DIM: usize = 32;
pub const XIANGQI_CONTEXT_DIM: usize = 12;
pub const XIANGQI_VOCABS: [usize; 9] = [198, 180, 360, 180, 360, 639, 694, 64, 18];
pub const XIANGQI_FEATURE_VERSION: &str = "xiangqi_raw_v01";
pub const XP: u8 = 0;
pub const XH: u8 = 1;
pub const XR: u8 = 2;
pub const XC: u8 = 3;
pub const XA: u8 = 4;
pub const XE: u8 = 5;
pub const XK: u8 = 6;
pub const XRED: u8 = 0;
pub const XBLACK: u8 = 1;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XPiece {
    pub kind: u8,
    pub color: u8,
}
#[derive(Debug, Clone)]
pub struct XiangqiBoard {
    pub sq: [Option<XPiece>; 90],
    pub stm: u8,
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
    f >= 0 && f < 9 && r >= 0 && r < 10
}
fn in_palace(f: i32, r: i32, color: u8) -> bool {
    if f < 3 || f > 5 {
        return false;
    }
    if color == XRED {
        (0..=2).contains(&r)
    } else {
        (7..=9).contains(&r)
    }
}
fn crossed(r: i32, color: u8) -> bool {
    if color == XRED {
        r >= 5
    } else {
        r <= 4
    }
}
fn board_kind(c: char) -> Option<u8> {
    match c {
        'p' => Some(XP),
        'h' => Some(XH),
        'r' => Some(XR),
        'c' => Some(XC),
        'a' => Some(XA),
        'e' => Some(XE),
        'k' => Some(XK),
        _ => None,
    }
}
impl XiangqiBoard {
    pub fn parse_fen(fen: &str) -> Result<XiangqiBoard> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.is_empty() {
            return Err(RuntimeError::BadFen("xiangqi: empty".to_string()));
        }
        let ranks: Vec<&str> = parts[0].split('/').collect();
        if ranks.len() != 10 {
            return Err(RuntimeError::BadFen("xiangqi: need 10 ranks".to_string()));
        }
        let mut sq: [Option<XPiece>; 90] = [None; 90];
        for (ri, rank) in ranks.iter().enumerate() {
            let r = 9 - ri as i32;
            let mut f: i32 = 0;
            for c in rank.chars() {
                if c.is_ascii_digit() {
                    f += c.to_digit(10).unwrap() as i32;
                } else {
                    let color = if c.is_ascii_uppercase() { XRED } else { XBLACK };
                    let kind = board_kind(c.to_ascii_lowercase())
                        .ok_or_else(|| RuntimeError::BadFen("xiangqi: bad piece".to_string()))?;
                    if f >= 9 {
                        return Err(RuntimeError::BadFen("xiangqi: rank overflow".to_string()));
                    }
                    sq[(r * 9 + f) as usize] = Some(XPiece { kind, color });
                    f += 1;
                }
            }
            if f != 9 {
                return Err(RuntimeError::BadFen("xiangqi: rank width".to_string()));
            }
        }
        let stm = if parts.len() > 1 {
            match parts[1] {
                "w" | "r" => XRED,
                "b" => XBLACK,
                _ => return Err(RuntimeError::BadFen("xiangqi: bad side".to_string())),
            }
        } else {
            XRED
        };
        let move_no = if parts.len() > 5 {
            parts[5].parse().unwrap_or(1).max(1)
        } else {
            1
        };
        Ok(XiangqiBoard { sq, stm, move_no })
    }
    pub fn piece_count(&self) -> u32 {
        self.sq.iter().filter(|c| c.is_some()).count() as u32
    }
}
pub fn game_phase(board: &XiangqiBoard) -> u8 {
    let mut n = 0;
    for c in board.sq.iter().flatten() {
        if matches!(c.kind, XA | XE | XH | XR | XC) {
            n += 1;
        }
    }
    if n >= 14 {
        0
    } else if n >= 8 {
        1
    } else {
        2
    }
}
fn sliding_clear(b: &XiangqiBoard, frm: usize, target: usize, df: i32, dr: i32) -> bool {
    let (mut f, mut r) = (sq_file(frm) + df, sq_rank(frm) + dr);
    while (f, r) != (sq_file(target), sq_rank(target)) {
        if b.sq[make_sq(f, r)].is_some() {
            return false;
        }
        f += df;
        r += dr;
    }
    true
}
pub fn piece_attacks(b: &XiangqiBoard, frm: usize, target: usize) -> bool {
    if frm == target {
        return false;
    }
    let (kind, color) = match b.sq[frm] {
        Some(p) => (p.kind, p.color),
        None => return false,
    };
    let (ff, rf) = (sq_file(frm), sq_rank(frm));
    let (tf, rt) = (sq_file(target), sq_rank(target));
    let (df, dr) = (tf - ff, rt - rf);
    let (adf, adr) = (df.abs(), dr.abs());
    let fwd = if color == XRED { 1 } else { -1 };
    match kind {
        XK => {
            if adf + adr == 1 && in_palace(tf, rt, color) {
                return true;
            }
            match b.sq[target] {
                Some(t) if t.kind == XK => {
                    if tf != ff {
                        return false;
                    }
                    let (lo, hi) = if frm < target { (frm, target) } else { (target, frm) };
                    let mut s = lo + 9;
                    while s < hi {
                        if b.sq[s].is_some() {
                            return false;
                        }
                        s += 9;
                    }
                    true
                }
                _ => false,
            }
        }
        XA => adf == 1 && adr == 1 && in_palace(tf, rt, color),
        XE => {
            if adf != 2 || adr != 2 {
                return false;
            }
            if color == XRED && rt > 4 {
                return false;
            }
            if color == XBLACK && rt < 5 {
                return false;
            }
            b.sq[make_sq(ff + df / 2, rf + dr / 2)].is_none()
        }
        XH => {
            if (adf, adr) != (1, 2) && (adf, adr) != (2, 1) {
                return false;
            }
            let leg = if adf == 2 {
                make_sq(ff + df / 2, rf)
            } else {
                make_sq(ff, rf + dr / 2)
            };
            b.sq[leg].is_none()
        }
        XR => {
            if df != 0 && dr != 0 {
                return false;
            }
            let sf = if df == 0 { 0 } else if df > 0 { 1 } else { -1 };
            let sr = if dr == 0 { 0 } else if dr > 0 { 1 } else { -1 };
            sliding_clear(b, frm, target, sf, sr)
        }
        XC => {
            if df != 0 && dr != 0 {
                return false;
            }
            let sf = if df == 0 { 0 } else if df > 0 { 1 } else { -1 };
            let sr = if dr == 0 { 0 } else if dr > 0 { 1 } else { -1 };
            let (mut f, mut r) = (ff + sf, rf + sr);
            let mut screens = 0;
            while (f, r) != (tf, rt) {
                if b.sq[make_sq(f, r)].is_some() {
                    screens += 1;
                }
                f += sf;
                r += sr;
            }
            screens == 1
        }
        XP => {
            if df == 0 && dr == fwd {
                return true;
            }
            dr == 0 && adf == 1 && crossed(rf, color)
        }
        _ => false,
    }
}
pub fn flying(b: &XiangqiBoard) -> bool {
    let mut kings = [90, 90];
    for sq in 0..90 {
        match b.sq[sq] {
            Some(c) if c.kind == XK => kings[c.color as usize] = sq,
            _ => {}
        }
    }
    if kings[0] >= 90 || kings[1] >= 90 {
        return false;
    }
    let (a, bb) = (kings[0], kings[1]);
    if sq_file(a) != sq_file(bb) {
        return false;
    }
    let (lo, hi) = if a < bb { (a, bb) } else { (bb, a) };
    let mut s = lo + 9;
    while s < hi {
        if b.sq[s].is_some() {
            return false;
        }
        s += 9;
    }
    true
}
pub fn extract_features(b: &XiangqiBoard) -> Vec<(u8, u16)> {
    let mut feats: Vec<(u8, u16)> = Vec::new();
    let us = b.stm;
    for sq in 0..90 {
        let cell = match b.sq[sq] {
            Some(c) => c,
            None => continue,
        };
        let ci = if cell.color == us { 0 } else { 1 } as u16;
        match cell.kind {
            XP => {
                feats.push((0, ci * 90 + sq as u16));
                if crossed(sq_rank(sq), cell.color) {
                    feats.push((0, (180 + ci * 9 + sq_file(sq) as u16) as u16));
                }
            }
            XK => feats.push((1, ci * 90 + sq as u16)),
            XA | XE => feats.push((2, ci * 2 * 90 + (cell.kind - XA) as u16 * 90 + sq as u16)),
            XH => feats.push((3, ci * 90 + sq as u16)),
            XR | XC => feats.push((4, ci * 2 * 90 + (cell.kind - XR) as u16 * 90 + sq as u16)),
            _ => {}
        }
        match cell.kind {
            XP | XH | XR | XC | XA | XE | XK => {
                feats.push((6, cell.kind as u16 * 90 + sq as u16))
            }
            _ => {}
        }
    }
    let fly = flying(b);
    for vsq in 0..90 {
        let victim = match b.sq[vsq] {
            Some(c) => c,
            None => continue,
        };
        for asq in 0..90 {
            let attacker = match b.sq[asq] {
                Some(c) => c,
                None => continue,
            };
            if attacker.color == victim.color {
                continue;
            }
            if !piece_attacks(b, asq, vsq) {
                continue;
            }
            feats.push((5, victim.kind as u16 * 90 + vsq as u16));
            feats.push((5, (630 + sq_file(asq) as u16) as u16));
        }
    }
    for f in 0..9 {
        let mut majors = 0;
        let mut majors_t = 0;
        for r in 0..10 {
            match b.sq[make_sq(f, r)] {
                Some(c) if matches!(c.kind, XH | XR | XC) && c.color == us => majors += 1,
                Some(c) if matches!(c.kind, XH | XR | XC) && c.color != us => majors_t += 1,
                _ => {}
            }
        }
        if majors >= 2 {
            feats.push((8, f as u16));
        }
        if majors_t >= 2 {
            feats.push((8, (9 + f) as u16));
        }
    }
    feats.push((7, us as u16));
    feats.push((7, (2 + game_phase(b) as u16) as u16));
    let total = b.piece_count();
    feats.push((7, (5 + ((32 - total) / 2).min(15)) as u16));
    let mut kus = 90;
    for sq in 0..90 {
        match b.sq[sq] {
            Some(c) if c.kind == XK && c.color == us => {
                kus = sq;
                break;
            }
            _ => {}
        }
    }
    let mut check = 0;
    if kus < 90 {
        for asq in 0..90 {
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
        if check == 0 && fly {
            check = 1;
        }
    }
    if check == 1 {
        feats.push((7, 21));
    }
    if fly {
        feats.push((7, 22));
    }
    feats.push((7, (23 + (b.move_no / 20).min(5)) as u16));
    feats.sort_unstable();
    feats.dedup();
    feats
}
pub fn compute_context(b: &XiangqiBoard) -> Vec<f32> {
    let us = b.stm;
    let mut us_majors = 0u32;
    let mut them_majors = 0u32;
    let mut us_pawns = 0u32;
    let mut them_pawns = 0u32;
    let mut us_crossed = 0u32;
    let mut us_palace = 0u32;
    let mut us_total = 0u32;
    let mut them_total = 0u32;
    let mut kus = 90;
    for sq in 0..90 {
        let cell = match b.sq[sq] {
            Some(c) => c,
            None => continue,
        };
        if cell.color == us {
            us_total += 1;
            if matches!(cell.kind, XH | XR | XC) {
                us_majors += 1;
            }
            if cell.kind == XP {
                us_pawns += 1;
                if crossed(sq_rank(sq), cell.color) {
                    us_crossed += 1;
                }
            }
            if cell.kind == XA || cell.kind == XE {
                us_palace += 1;
            }
            if cell.kind == XK {
                kus = sq;
            }
        } else {
            them_total += 1;
            if matches!(cell.kind, XH | XR | XC) {
                them_majors += 1;
            }
            if cell.kind == XP {
                them_pawns += 1;
            }
        }
    }
    let mut check = 0;
    if kus < 90 {
        for asq in 0..90 {
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
        if check == 0 && flying(b) {
            check = 1;
        }
    }
    let fly = if flying(b) { 1 } else { 0 };
    let lead = us_total as f32 - them_total as f32;
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
        clip(game_phase(b) as f32 / 2.0),
        clip(us_majors as f32 / 12.0),
        clip(them_majors as f32 / 12.0),
        clip(us_pawns as f32 / 5.0),
        clip(them_pawns as f32 / 5.0),
        clip(us_crossed as f32 / 5.0),
        clip(check as f32),
        clip(fly as f32),
        clip(us_palace as f32 / 4.0),
        clip((lead + 16.0) / 32.0),
        clip(b.move_no as f32 / 200.0),
    ]
}
