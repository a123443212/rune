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
pub const PAWN: u8 = 1;
pub const KNIGHT: u8 = 2;
pub const BISHOP: u8 = 3;
pub const ROOK: u8 = 4;
pub const QUEEN: u8 = 5;
pub const KING: u8 = 6;
pub const WHITE: u8 = 0;
pub const BLACK: u8 = 1;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub kind: u8,
    pub color: u8,
}
#[derive(Debug, Clone)]
struct Undo {
    from: usize,
    to: usize,
    moved: Piece,
    captured: Option<Piece>,
    cap_sq: usize,
    prev_castle: u8,
    prev_ep: i16,
    prev_halfmove: u16,
    prev_fullmove: u16,
    promo: u8,
    was_castle: bool,
    was_ep: bool,
}
#[derive(Debug, Clone)]
pub struct Board {
    pub sq: [Option<Piece>; 64],
    pub stm: u8,
    pub castle_mask: u8,
    pub ep_sq: i16,
    pub halfmove_clock: u16,
    pub fullmove_number: u16,
    stack: Vec<Undo>,
}
pub fn sq_file(sq: usize) -> i32 {
    (sq & 7) as i32
}
pub fn sq_rank(sq: usize) -> i32 {
    (sq >> 3) as i32
}
pub fn make_sq(f: i32, r: i32) -> usize {
    (r * 8 + f) as usize
}
pub fn on_board(f: i32, r: i32) -> bool {
    f >= 0 && f < 8 && r >= 0 && r < 8
}
fn parse_piece(c: char) -> Option<(u8, u8)> {
    let color = if c.is_ascii_uppercase() { WHITE } else { BLACK };
    let kind = match c.to_ascii_lowercase() {
        'p' => PAWN,
        'n' => KNIGHT,
        'b' => BISHOP,
        'r' => ROOK,
        'q' => QUEEN,
        'k' => KING,
        _ => return None,
    };
    Some((kind, color))
}
fn sq_name(sq: usize) -> String {
    let f = (sq & 7) as u8 + b'a';
    let r = (sq >> 3) as u8 + b'1';
    String::from_utf8(vec![f, r]).unwrap_or_default()
}
fn parse_sq(s: &str) -> Result<usize> {
    let b = s.as_bytes();
    if b.len() != 2 {
        return Err(RuntimeError::BadUci(s.to_string()));
    }
    if !(b[0] >= b'a' && b[0] <= b'h') {
        return Err(RuntimeError::BadUci(s.to_string()));
    }
    if !(b[1] >= b'1' && b[1] <= b'8') {
        return Err(RuntimeError::BadUci(s.to_string()));
    }
    Ok(((b[1] - b'1') as usize) * 8 + ((b[0] - b'a') as usize))
}
impl Board {
    pub fn startpos() -> Board {
        Board::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap_or_else(|_| Board {
            sq: [None; 64],
            stm: WHITE,
            castle_mask: 0,
            ep_sq: -1,
            halfmove_clock: 0,
            fullmove_number: 1,
            stack: Vec::new(),
        })
    }
    pub fn parse_fen(fen: &str) -> Result<Board> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(RuntimeError::BadFen("need 4 fields".to_string()));
        }
        let mut sq: [Option<Piece>; 64] = [None; 64];
        let mut rank: i32 = 7;
        let mut file: i32 = 0;
        for c in parts[0].chars() {
            if c == '/' {
                if file != 8 {
                    return Err(RuntimeError::BadFen("short rank".to_string()));
                }
                rank -= 1;
                file = 0;
                if rank < 0 {
                    return Err(RuntimeError::BadFen("too many ranks".to_string()));
                }
            } else if c.is_ascii_digit() {
                file += c.to_digit(10).unwrap() as i32;
                if file > 8 {
                    return Err(RuntimeError::BadFen("rank overflow".to_string()));
                }
            } else if let Some((kind, color)) = parse_piece(c) {
                if file >= 8 || rank < 0 {
                    return Err(RuntimeError::BadFen("square overflow".to_string()));
                }
                sq[make_sq(file, rank)] = Some(Piece { kind, color });
                file += 1;
            } else {
                return Err(RuntimeError::BadFen("bad char".to_string()));
            }
        }
        if rank != 0 || file != 8 {
            return Err(RuntimeError::BadFen("bad placement".to_string()));
        }
        let stm = match parts[1] {
            "w" => WHITE,
            "b" => BLACK,
            _ => return Err(RuntimeError::BadFen("bad side".to_string())),
        };
        let mut mask: u8 = 0;
        if parts[2] != "-" {
            for c in parts[2].chars() {
                match c {
                    'K' => mask |= 1,
                    'Q' => mask |= 2,
                    'k' => mask |= 4,
                    'q' => mask |= 8,
                    _ => return Err(RuntimeError::BadFen("bad castling".to_string())),
                }
            }
        }
        let ep_sq: i16 = if parts[3] == "-" {
            -1
        } else {
            parse_sq(parts[3])? as i16
        };
        let halfmove_clock: u16 = match parts.get(4) {
            None => 0,
            Some(s) => s.parse().map_err(|_| RuntimeError::BadFen("bad halfmove".to_string()))?,
        };
        let fullmove_number: u16 = match parts.get(5) {
            None => 1,
            Some(s) => s.parse().map_err(|_| RuntimeError::BadFen("bad fullmove".to_string()))?,
        };
        if fullmove_number == 0 {
            return Err(RuntimeError::BadFen("bad fullmove".to_string()));
        }
        Ok(Board { sq, stm, castle_mask: mask, ep_sq, halfmove_clock, fullmove_number, stack: Vec::new() })
    }
    pub fn piece_count(&self) -> usize {
        self.sq.iter().filter(|c| c.is_some()).count()
    }
    pub fn game_phase(&self) -> u8 {
        let n = self.sq.iter().filter(|c| matches!(c, Some(p) if p.kind != PAWN && p.kind != KING)).count();
        if n >= 12 {
            0
        } else if n >= 6 {
            1
        } else {
            2
        }
    }
    pub fn to_fen(&self) -> String {
        let mut s = String::new();
        for r in (0..8).rev() {
            let mut empty = 0;
            for f in 0..8 {
                match self.sq[make_sq(f, r)] {
                    None => empty += 1,
                    Some(p) => {
                        if empty > 0 {
                            s.push_str(&empty.to_string());
                            empty = 0;
                        }
                        let mut c = match p.kind {
                            PAWN => 'p',
                            KNIGHT => 'n',
                            BISHOP => 'b',
                            ROOK => 'r',
                            QUEEN => 'q',
                            _ => 'k',
                        };
                        if p.color == WHITE {
                            c = c.to_ascii_uppercase();
                        }
                        s.push(c);
                    }
                }
            }
            if empty > 0 {
                s.push_str(&empty.to_string());
            }
            if r > 0 {
                s.push('/');
            }
        }
        s.push(' ');
        s.push(if self.stm == WHITE { 'w' } else { 'b' });
        s.push(' ');
        if self.castle_mask == 0 {
            s.push('-');
        } else {
            if self.castle_mask & 1 != 0 {
                s.push('K');
            }
            if self.castle_mask & 2 != 0 {
                s.push('Q');
            }
            if self.castle_mask & 4 != 0 {
                s.push('k');
            }
            if self.castle_mask & 8 != 0 {
                s.push('q');
            }
        }
        s.push(' ');
        if self.ep_sq < 0 {
            s.push('-');
        } else {
            s.push_str(&sq_name(self.ep_sq as usize));
        }
        s.push(' ');
        s.push_str(&self.halfmove_clock.to_string());
        s.push(' ');
        s.push_str(&self.fullmove_number.to_string());
        s
    }
    pub fn apply_uci(&mut self, uci: &str) -> Result<()> {
        let u = uci.trim();
        if u.len() < 4 {
            return Err(RuntimeError::BadUci(u.to_string()));
        }
        let from = parse_sq(&u[0..2])?;
        let to = parse_sq(&u[2..4])?;
        let mut promo: u8 = 0;
        if u.len() >= 5 {
            promo = match u[4..5].to_ascii_lowercase().as_str() {
                "n" => KNIGHT,
                "b" => BISHOP,
                "r" => ROOK,
                "q" => QUEEN,
                _ => return Err(RuntimeError::BadUci(u.to_string())),
            };
        }
        let moved = self.sq[from].ok_or_else(|| RuntimeError::BadUci("empty from".to_string()))?;
        let mut cap_sq = to;
        let mut captured = self.sq[to];
        let mut was_ep = false;
        if moved.kind == PAWN && to as i16 == self.ep_sq && captured.is_none() {
            was_ep = true;
            let dir = if moved.color == WHITE { -1 } else { 1 };
            let f = sq_file(to);
            let r = sq_rank(to) + dir;
            cap_sq = make_sq(f, r);
            captured = self.sq[cap_sq];
        }
        let mut was_castle = false;
        if moved.kind == KING {
            let df = sq_file(to) - sq_file(from);
            if df.abs() == 2 {
                was_castle = true;
            }
        }
        let undo = Undo {
            from,
            to,
            moved,
            captured,
            cap_sq,
            prev_castle: self.castle_mask,
            prev_ep: self.ep_sq,
            prev_halfmove: self.halfmove_clock,
            prev_fullmove: self.fullmove_number,
            promo,
            was_castle,
            was_ep,
        };
        self.sq[from] = None;
        if was_ep {
            self.sq[cap_sq] = None;
        }
        let placed = if promo != 0 {
            Piece { kind: promo, color: moved.color }
        } else {
            moved
        };
        self.sq[to] = Some(placed);
        if was_castle {
            let r = sq_rank(from);
            if sq_file(to) == 6 {
                let rf = make_sq(7, r);
                let rt = make_sq(5, r);
                self.sq[rt] = self.sq[rf];
                self.sq[rf] = None;
            } else {
                let rf = make_sq(0, r);
                let rt = make_sq(3, r);
                self.sq[rt] = self.sq[rf];
                self.sq[rf] = None;
            }
        }
        self.ep_sq = -1;
        if moved.kind == PAWN {
            let dr = sq_rank(to) - sq_rank(from);
            if dr.abs() == 2 {
                let r = (sq_rank(from) + sq_rank(to)) / 2;
                self.ep_sq = make_sq(sq_file(from), r) as i16;
            }
        }
        if moved.kind == KING {
            if moved.color == WHITE {
                self.castle_mask &= !(1 | 2);
            } else {
                self.castle_mask &= !(4 | 8);
            }
        }
        if moved.kind == ROOK {
            if from == make_sq(0, 0) {
                self.castle_mask &= !2;
            }
            if from == make_sq(7, 0) {
                self.castle_mask &= !1;
            }
            if from == make_sq(0, 7) {
                self.castle_mask &= !8;
            }
            if from == make_sq(7, 7) {
                self.castle_mask &= !4;
            }
        }
        if let Some(cp) = captured {
            if cp.kind == ROOK {
                if cap_sq == make_sq(0, 0) {
                    self.castle_mask &= !2;
                }
                if cap_sq == make_sq(7, 0) {
                    self.castle_mask &= !1;
                }
                if cap_sq == make_sq(0, 7) {
                    self.castle_mask &= !8;
                }
                if cap_sq == make_sq(7, 7) {
                    self.castle_mask &= !4;
                }
            }
        }
        let _ = undo.was_ep;
        if moved.kind == PAWN || captured.is_some() {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock = self.halfmove_clock.saturating_add(1);
        }
        if moved.color == BLACK {
            self.fullmove_number = self.fullmove_number.saturating_add(1);
        }
        self.stack.push(undo);
        self.stm = 1 - self.stm;
        Ok(())
    }
    pub fn unmake(&mut self) -> Result<()> {
        let u = self.stack.pop().ok_or_else(|| RuntimeError::InvalidState("empty stack".to_string()))?;
        self.stm = 1 - self.stm;
        self.castle_mask = u.prev_castle;
        self.ep_sq = u.prev_ep;
        self.halfmove_clock = u.prev_halfmove;
        self.fullmove_number = u.prev_fullmove;
        if u.was_castle {
            let r = sq_rank(u.from);
            if sq_file(u.to) == 6 {
                let rf = make_sq(7, r);
                let rt = make_sq(5, r);
                self.sq[rf] = self.sq[rt];
                self.sq[rt] = None;
            } else {
                let rf = make_sq(0, r);
                let rt = make_sq(3, r);
                self.sq[rf] = self.sq[rt];
                self.sq[rt] = None;
            }
        }
        self.sq[u.from] = Some(u.moved);
        self.sq[u.to] = None;
        if u.was_ep {
            self.sq[u.cap_sq] = u.captured;
        } else if let Some(c) = u.captured {
            self.sq[u.cap_sq] = Some(c);
        }
        let _ = u.promo;
        let _ = u.was_castle;
        Ok(())
    }
    pub fn stack_len(&self) -> usize {
        self.stack.len()
    }
}
