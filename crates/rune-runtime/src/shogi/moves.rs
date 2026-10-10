use super::{make_sq, on_board, sq_file, sq_rank, slide_dirs, step_moves, ShogiBoard, B, DR, G, HB, L, N, P, PN, PL, PP, PS, R, S, SBLACK};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SMove {
    Normal { fr: usize, to: usize, promo: bool },
    Drop { to: usize, piece: u8 },
}

pub fn promotable(kind: u8) -> bool {
    matches!(kind, P | L | N | S | B | R)
}

pub fn unpromote(kind: u8) -> u8 {
    match kind {
        PP => P,
        PL => L,
        PN => N,
        PS => S,
        HB => B,
        DR => R,
        _ => kind,
    }
}

pub fn hand_index(kind: u8) -> Option<usize> {
    match kind {
        P => Some(0),
        L => Some(1),
        N => Some(2),
        S => Some(3),
        G => Some(4),
        B => Some(5),
        R => Some(6),
        _ => None,
    }
}

fn in_zone(rank: i32, color: u8) -> bool {
    if color == SBLACK {
        rank <= 2
    } else {
        rank >= 6
    }
}

fn must_promote(kind: u8, color: u8, to_rank: i32) -> bool {
    let last = if color == SBLACK { 0 } else { 8 };
    if kind == P || kind == L {
        return to_rank == last;
    }
    if kind == N {
        if color == SBLACK {
            return to_rank <= 1;
        }
        return to_rank >= 7;
    }
    false
}

fn promo_options(b: &ShogiBoard, frm: usize, to: usize, color: u8) -> Vec<bool> {
    let kind = match b.sq[frm] {
        Some(p) => p.kind,
        None => return Vec::new(),
    };
    if !promotable(kind) {
        return vec![false];
    }
    if in_zone(sq_rank(frm), color) || in_zone(sq_rank(to), color) {
        if must_promote(kind, color, sq_rank(to)) {
            return vec![true];
        }
        return vec![false, true];
    }
    vec![false]
}

fn step_dests(b: &ShogiBoard, frm: usize, color: u8) -> Vec<usize> {
    let kind = match b.sq[frm] {
        Some(p) => p.kind,
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    for (df, dr) in step_moves(kind, color) {
        let tf = sq_file(frm) + df;
        let tr = sq_rank(frm) + dr;
        if on_board(tf, tr) {
            out.push(make_sq(tf, tr));
        }
    }
    if kind == L {
        let f: i32 = if color == SBLACK { -1 } else { 1 };
        let tf = sq_file(frm);
        let mut tr = sq_rank(frm) + f;
        while on_board(tf, tr) {
            let t = make_sq(tf, tr);
            out.push(t);
            if b.sq[t].is_some() {
                break;
            }
            tr += f;
        }
    }
    for (df, dr) in slide_dirs(kind) {
        let (mut tf, mut tr) = (sq_file(frm) + df, sq_rank(frm) + dr);
        while on_board(tf, tr) {
            let t = make_sq(tf, tr);
            out.push(t);
            if b.sq[t].is_some() {
                break;
            }
            tf += df;
            tr += dr;
        }
    }
    out
}

fn drop_squares(b: &ShogiBoard, count: u8, piece: u8, color: u8) -> Vec<usize> {
    let mut out = Vec::new();
    if count == 0 {
        return out;
    }
    for sq in 0..81 {
        if b.sq[sq].is_some() {
            continue;
        }
        let r = sq_rank(sq);
        let f = sq_file(sq);
        if piece == P || piece == L {
            if (color == SBLACK && r == 0) || (color != SBLACK && r == 8) {
                continue;
            }
        }
        if piece == N {
            if (color == SBLACK && r <= 1) || (color != SBLACK && r >= 7) {
                continue;
            }
        }
        if piece == P {
            let mut bad = false;
            for rr in 0..9 {
                match b.sq[make_sq(f, rr)] {
                    Some(p) if p.kind == P && p.color == color => {
                        bad = true;
                        break;
                    }
                    _ => {}
                }
            }
            if bad {
                continue;
            }
        }
        out.push(sq);
    }
    out
}

const HAND_ORDER: [u8; 7] = [P, L, N, S, G, B, R];

pub fn pseudo_moves(b: &ShogiBoard) -> Vec<SMove> {
    let mut out = Vec::new();
    for frm in 0..81 {
        let own = match b.sq[frm] {
            Some(p) if p.color == b.stm => true,
            _ => false,
        };
        if !own {
            continue;
        }
        for to in step_dests(b, frm, b.stm) {
            match b.sq[to] {
                Some(p) if p.color == b.stm => continue,
                _ => {}
            }
            for promo in promo_options(b, frm, to, b.stm) {
                out.push(SMove::Normal { fr: frm, to, promo });
            }
        }
    }
    for piece in HAND_ORDER {
        let ti = hand_index(piece).unwrap_or(0);
        let n = b.hand[b.stm as usize][ti];
        for to in drop_squares(b, n, piece, b.stm) {
            out.push(SMove::Drop { to, piece });
        }
    }
    out
}
