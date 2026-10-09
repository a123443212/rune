use crate::error::{Result, RuntimeError};

pub const GO_VOCABS: [usize; 9] = [361, 361, 361, 64, 64, 361, 361, 64, 18];
pub const GO_CONTEXT_DIM: usize = 12;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoBoard {
    pub size: usize,
    pub stones: Vec<i8>,
    pub stm: u8,
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
        if size != 9 && size != 13 && size != 19 {
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

    pub fn liberties_of(&self, r: usize, c: usize) -> usize {
        let target = self.at(r, c);
        if target == 0 {
            return 0;
        }
        let n = self.size;
        let mut seen = vec![false; n * n];
        let mut stack = vec![(r, c)];
        seen[r * n + c] = true;
        let mut libs = std::collections::HashSet::new();
        while let Some((cr, cc)) = stack.pop() {
            for (dr, dc) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nr = cr as i32 + dr;
                let nc = cc as i32 + dc;
                if nr < 0 || nc < 0 || nr >= n as i32 || nc >= n as i32 {
                    continue;
                }
                let ur = nr as usize;
                let uc = nc as usize;
                let v = self.at(ur, uc);
                if v == 0 {
                    libs.insert(ur * n + uc);
                } else if v == target && !seen[ur * n + uc] {
                    seen[ur * n + uc] = true;
                    stack.push((ur, uc));
                }
            }
        }
        libs.len().min(8)
    }
}

pub fn extract_planes(b: &GoBoard) -> Vec<f32> {
    let n = b.size;
    let mut out = vec![0.0f32; n * n];
    for r in 0..n {
        for c in 0..n {
            let v = b.at(r, c);
            let rel: f32 = if v == 0 { 0.0 } else if (v == 1 && b.stm == 0) || (v == -1 && b.stm == 1) { 1.0 } else { -1.0 };
            out[r * n + c] = rel;
        }
    }
    out
}

pub fn compute_context(b: &GoBoard) -> Vec<f32> {
    let mut us = 0usize;
    let mut them = 0usize;
    for v in b.stones.iter() {
        if *v == 0 {
            continue;
        }
        let mine = (*v == 1 && b.stm == 0) || (*v == -1 && b.stm == 1);
        if mine {
            us += 1;
        } else {
            them += 1;
        }
    }
    let total = (b.size * b.size) as f32;
    vec![b.stm as f32, (us as f32 / total).clamp(0.0, 1.0), (them as f32 / total).clamp(0.0, 1.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
