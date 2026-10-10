use crate::error::{Result, RuntimeError};
use super::board::{clamp01_f32, liberty_map, valid_size, GoBoard};

pub const DEFAULT_KOMI: f32 = 7.5;

#[derive(Debug, Clone, PartialEq)]
pub struct GoState {
    pub board: GoBoard,
    pub ko: Option<usize>,
    pub komi: f32,
    pub move_no: u32,
    pub pass_no: u32,
}

fn parse_komi(tok: &str) -> Result<f32> {
    match tok.parse::<f32>() {
        Ok(v) => {
            if v < -50.0 || v > 50.0 {
                return Err(RuntimeError::InvalidState("bad go komi".to_string()));
            }
            Ok(v)
        }
        Err(_) => Err(RuntimeError::InvalidState("bad go komi".to_string())),
    }
}

fn parse_sq(tok: &str, n: usize) -> Result<Option<usize>> {
    if tok == "-" {
        return Ok(None);
    }
    match tok.parse::<usize>() {
        Ok(v) => {
            if v >= n * n {
                return Err(RuntimeError::InvalidState("bad go ko".to_string()));
            }
            Ok(Some(v))
        }
        Err(_) => Err(RuntimeError::InvalidState("bad go ko".to_string())),
    }
}

fn parse_count(tok: &str, lo: u32, hi: u32) -> Result<u32> {
    match tok.parse::<u32>() {
        Ok(v) => {
            if v < lo || v > hi {
                return Err(RuntimeError::InvalidState("bad go count".to_string()));
            }
            Ok(v)
        }
        Err(_) => Err(RuntimeError::InvalidState("bad go count".to_string())),
    }
}

pub fn komi_bucket(komi: f32) -> usize {
    let v = clamp01_f32((komi + 5.0) / 20.0);
    let b = (v * 7.99) as usize;
    if b > 7 {
        return 7;
    }
    b
}

pub fn move_bucket(move_no: u32) -> usize {
    let b = move_no.saturating_sub(1) / 20;
    if b > 5 {
        return 5;
    }
    b as usize
}

pub fn game_phase(move_no: u32, n: usize) -> u8 {
    let total = (n * n) as u32;
    if move_no <= total / 3 {
        return 0;
    }
    if move_no <= (2 * total) / 3 {
        return 1;
    }
    2
}

impl GoState {
    pub fn parse(s: &str) -> Result<GoState> {
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.is_empty() {
            return Err(RuntimeError::InvalidState("empty go state".to_string()));
        }
        let rows: Vec<&str> = parts[0].split('/').collect();
        let n = rows.len();
        if !valid_size(n) {
            return Err(RuntimeError::InvalidState("bad go size".to_string()));
        }
        let board = GoBoard::parse(s)?;
        let mut size_board = board;
        size_board.size = n;
        let mut ko: Option<usize> = None;
        let mut komi: f32 = DEFAULT_KOMI;
        let mut move_no: u32 = 1;
        let mut pass_no: u32 = 0;
        if parts.len() > 2 {
            ko = parse_sq(parts[2], n)?;
        }
        if parts.len() > 3 {
            komi = parse_komi(parts[3])?;
        }
        if parts.len() > 4 {
            move_no = parse_count(parts[4], 1, 10000)?;
        }
        if parts.len() > 5 {
            pass_no = parse_count(parts[5], 0, 500)?;
        }
        Ok(GoState { board: size_board, ko, komi, move_no, pass_no })
    }

    pub fn size(&self) -> usize {
        self.board.size
    }

    pub fn stm(&self) -> u8 {
        self.board.stm
    }

    pub fn encode(&self) -> String {
        let n = self.board.size;
        let mut rows = Vec::new();
        for r in 0..n {
            let mut row = String::new();
            for c in 0..n {
                let v = self.board.at(r, c);
                if v == 0 {
                    row.push('.');
                } else if v == 1 {
                    row.push('X');
                } else {
                    row.push('O');
                }
            }
            rows.push(row);
        }
        let grid = rows.join("/");
        let s = if self.board.stm == 0 { "b" } else { "w" };
        let k = match self.ko {
            Some(v) => v.to_string(),
            None => "-".to_string(),
        };
        format!("{} {} {} {} {} {}", grid, s, k, self.komi, self.move_no, self.pass_no)
    }

    pub fn normalize(&self) -> String {
        let n = self.board.size;
        let mut rows = Vec::new();
        for r in 0..n {
            let mut row = String::new();
            for c in 0..n {
                let v = self.board.at(r, c);
                if v == 0 {
                    row.push('.');
                } else if v == 1 {
                    row.push('X');
                } else {
                    row.push('O');
                }
            }
            rows.push(row);
        }
        let grid = rows.join("/");
        let s = if self.board.stm == 0 { "b" } else { "w" };
        match self.ko {
            Some(v) => format!("{} {} {}", grid, s, v),
            None => format!("{} {}", grid, s),
        }
    }

    pub fn context_v02(&self) -> Vec<f32> {
        let n = self.board.size;
        let stm = self.board.stm;
        let mut us = 0usize;
        let mut them = 0usize;
        let mut empty = 0usize;
        for v in self.board.stones.iter() {
            if *v == 0 {
                empty += 1;
            } else {
                let mine = (*v == 1 && stm == 0) || (*v == -1 && stm == 1);
                if mine {
                    us += 1;
                } else {
                    them += 1;
                }
            }
        }
        let total = (n * n) as f32;
        let libs = liberty_map(&self.board.stones, n);
        let mut l1 = 0usize;
        let mut l2 = 0usize;
        let mut l3 = 0usize;
        for sq in 0..n * n {
            let v = self.board.stones[sq];
            if v == 0 {
                continue;
            }
            let mine = (v == 1 && stm == 0) || (v == -1 && stm == 1);
            if !mine {
                continue;
            }
            let k = libs[sq];
            if k <= 1 {
                l1 += 1;
            } else if k == 2 {
                l2 += 1;
            } else {
                l3 += 1;
            }
        }
        let (f1, f2, f3) = if us > 0 {
            (clamp01_f32(l1 as f32 / us as f32), clamp01_f32(l2 as f32 / us as f32), clamp01_f32(l3 as f32 / us as f32))
        } else {
            (0.0, 0.0, 0.0)
        };
        let ko_f = if self.ko.is_some() { 1.0 } else { 0.0 };
        vec![
            clamp01_f32(stm as f32),
            clamp01_f32(us as f32 / total),
            clamp01_f32(them as f32 / total),
            clamp01_f32(self.komi / 15.0),
            clamp01_f32(self.move_no as f32 / 200.0),
            clamp01_f32(self.pass_no as f32 / 2.0),
            clamp01_f32(ko_f),
            clamp01_f32(f1),
            clamp01_f32(f2),
            clamp01_f32(f3),
            clamp01_f32(empty as f32 / total),
            clamp01_f32(game_phase(self.move_no, n) as f32 / 2.0),
        ]
    }

    pub fn phase_v02(&self) -> u8 {
        game_phase(self.move_no, self.board.size)
    }
}
