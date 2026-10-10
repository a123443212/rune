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

use super::{make_sq, on_board, sq_file, sq_rank, XA, XC, XE, XH, XK, XP, XR, XRED, XiangqiBoard};

fn pawn_dests(_b: &XiangqiBoard, sq: usize, color: u8) -> Vec<usize> {
    let f = sq_file(sq);
    let r = sq_rank(sq);
    let fwd: i32 = if color == XRED { 1 } else { -1 };
    let mut out = Vec::new();
    if on_board(f, r + fwd) {
        out.push(make_sq(f, r + fwd));
    }
    let crossed = if color == XRED { r >= 5 } else { r <= 4 };
    if crossed {
        if on_board(f - 1, r) {
            out.push(make_sq(f - 1, r));
        }
        if on_board(f + 1, r) {
            out.push(make_sq(f + 1, r));
        }
    }
    out
}

fn horse_dests(b: &XiangqiBoard, sq: usize) -> Vec<usize> {
    let f = sq_file(sq);
    let r = sq_rank(sq);
    let mut out = Vec::new();
    let deltas: [(i32, i32); 8] = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];
    for (df, dr) in deltas {
        let tf = f + df;
        let tr = r + dr;
        if !on_board(tf, tr) {
            continue;
        }
        let leg = if df.abs() == 2 {
            make_sq(f + df / 2, r)
        } else {
            make_sq(f, r + dr / 2)
        };
        if b.sq[leg].is_some() {
            continue;
        }
        out.push(make_sq(tf, tr));
    }
    out
}

fn ray(b: &XiangqiBoard, sq: usize, df: i32, dr: i32, capture: bool) -> Vec<usize> {
    let mut out = Vec::new();
    let (mut f, mut r) = (sq_file(sq) + df, sq_rank(sq) + dr);
    while on_board(f, r) {
        let t = make_sq(f, r);
        if b.sq[t].is_none() {
            if !capture {
                out.push(t);
            }
        } else {
            if capture {
                out.push(t);
            }
            break;
        }
        f += df;
        r += dr;
    }
    out
}

fn rook_dests(b: &XiangqiBoard, sq: usize, color: u8) -> Vec<usize> {
    let mut out = Vec::new();
    let dirs: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    for (df, dr) in dirs {
        out.extend(ray(b, sq, df, dr, false));
        for t in ray(b, sq, df, dr, true) {
            match b.sq[t] {
                Some(p) if p.color != color => out.push(t),
                _ => {}
            }
        }
    }
    out
}

fn cannon_dests(b: &XiangqiBoard, sq: usize, color: u8) -> Vec<usize> {
    let f = sq_file(sq);
    let r = sq_rank(sq);
    let mut out = Vec::new();
    let dirs: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    for (df, dr) in dirs {
        let (mut cf, mut cr) = (f + df, r + dr);
        while on_board(cf, cr) && b.sq[make_sq(cf, cr)].is_none() {
            out.push(make_sq(cf, cr));
            cf += df;
            cr += dr;
        }
        if !on_board(cf, cr) {
            continue;
        }
        cf += df;
        cr += dr;
        while on_board(cf, cr) {
            let t = make_sq(cf, cr);
            match b.sq[t] {
                Some(p) => {
                    if p.color != color {
                        out.push(t);
                    }
                    break;
                }
                None => {}
            }
            cf += df;
            cr += dr;
        }
    }
    out
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

fn advisor_dests(sq: usize, color: u8) -> Vec<usize> {
    let f = sq_file(sq);
    let r = sq_rank(sq);
    let mut out = Vec::new();
    for (df, dr) in [(-1, -1), (1, -1), (-1, 1), (1, 1)] {
        if in_palace(f + df, r + dr, color) {
            out.push(make_sq(f + df, r + dr));
        }
    }
    out
}

fn elephant_dests(b: &XiangqiBoard, sq: usize, color: u8) -> Vec<usize> {
    let f = sq_file(sq);
    let r = sq_rank(sq);
    let mut out = Vec::new();
    for (df, dr) in [(-2, -2), (2, -2), (-2, 2), (2, 2)] {
        let tf = f + df;
        let tr = r + dr;
        if !on_board(tf, tr) {
            continue;
        }
        if color == XRED && tr > 4 {
            continue;
        }
        if color != XRED && tr < 5 {
            continue;
        }
        if b.sq[make_sq(f + df / 2, r + dr / 2)].is_some() {
            continue;
        }
        out.push(make_sq(tf, tr));
    }
    out
}

fn king_dests(b: &XiangqiBoard, sq: usize, color: u8) -> Vec<usize> {
    let f = sq_file(sq);
    let r = sq_rank(sq);
    let mut out = Vec::new();
    for (df, dr) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        if in_palace(f + df, r + dr, color) {
            out.push(make_sq(f + df, r + dr));
        }
    }
    let foe = 1 - color;
    for t in 0..90 {
        match b.sq[t] {
            Some(p) if p.kind == XK && p.color == foe => {
                if sq_file(t) == f {
                    let (lo, hi) = if sq < t { (sq, t) } else { (t, sq) };
                    let mut blocked = false;
                    let mut m = lo + 9;
                    while m < hi {
                        if b.sq[m].is_some() {
                            blocked = true;
                            break;
                        }
                        m += 9;
                    }
                    if !blocked {
                        out.push(t);
                    }
                }
            }
            _ => {}
        }
    }
    out
}

pub fn pseudo_dests(b: &XiangqiBoard, sq: usize) -> Vec<usize> {
    let (kind, color) = match b.sq[sq] {
        Some(p) => (p.kind, p.color),
        None => return Vec::new(),
    };
    match kind {
        XP => pawn_dests(b, sq, color),
        XH => horse_dests(b, sq),
        XR => rook_dests(b, sq, color),
        XC => cannon_dests(b, sq, color),
        XA => advisor_dests(sq, color),
        XE => elephant_dests(b, sq, color),
        XK => king_dests(b, sq, color),
        _ => Vec::new(),
    }
}
