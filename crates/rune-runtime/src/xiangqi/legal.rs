use crate::error::{Result, RuntimeError};
use super::moves::pseudo_dests;
use super::{piece_attacks, XiangqiBoard, XBLACK, XK, XRED};

pub fn encode_move(fr: usize, to: usize) -> String {
    format!("{:02}{:02}", fr, to)
}

pub fn decode_move(mv: &str) -> Result<(usize, usize)> {
    if mv.len() != 4 {
        return Err(RuntimeError::BadFen("xiangqi: bad move".to_string()));
    }
    let fr: usize = mv[0..2].parse().map_err(|_| RuntimeError::BadFen("xiangqi: bad move".to_string()))?;
    let to: usize = mv[2..4].parse().map_err(|_| RuntimeError::BadFen("xiangqi: bad move".to_string()))?;
    if fr >= 90 || to >= 90 {
        return Err(RuntimeError::BadFen("xiangqi: bad move".to_string()));
    }
    Ok((fr, to))
}

pub fn king_in_check(b: &XiangqiBoard, color: u8) -> bool {
    let mut kus: Option<usize> = None;
    for sq in 0..90 {
        match b.sq[sq] {
            Some(p) if p.kind == XK && p.color == color => {
                kus = Some(sq);
                break;
            }
            _ => {}
        }
    }
    let kus = match kus {
        Some(v) => v,
        None => return true,
    };
    for asq in 0..90 {
        match b.sq[asq] {
            Some(p) if p.color != color => {
                if piece_attacks(b, asq, kus) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

pub fn legal_moves(state: &str) -> Result<Vec<String>> {
    let b = XiangqiBoard::parse_fen(state)?;
    let mut out = Vec::new();
    for fr in 0..90 {
        let own = match b.sq[fr] {
            Some(p) if p.color == b.stm => true,
            _ => false,
        };
        if !own {
            continue;
        }
        for to in pseudo_dests(&b, fr) {
            match b.sq[to] {
                Some(p) if p.color == b.stm => continue,
                _ => {}
            }
            let mut nb = b.clone();
            nb.sq[to] = nb.sq[fr];
            nb.sq[fr] = None;
            if king_in_check(&nb, b.stm) {
                continue;
            }
            out.push(encode_move(fr, to));
        }
    }
    out.sort();
    Ok(out)
}

fn encode_board(b: &XiangqiBoard) -> String {
    let mut rows = Vec::new();
    for r in (0..10).rev() {
        let mut rank = String::new();
        let mut gap = 0;
        for f in 0..9 {
            match b.sq[r * 9 + f] {
                None => gap += 1,
                Some(p) => {
                    if gap > 0 {
                        rank.push_str(&gap.to_string());
                        gap = 0;
                    }
                    let ch = match p.kind {
                        0 => 'p',
                        1 => 'h',
                        2 => 'r',
                        3 => 'c',
                        4 => 'a',
                        5 => 'e',
                        _ => 'k',
                    };
                    if p.color == XRED {
                        rank.push(ch.to_ascii_uppercase());
                    } else {
                        rank.push(ch);
                    }
                }
            }
        }
        if gap > 0 {
            rank.push_str(&gap.to_string());
        }
        rows.push(rank);
    }
    rows.join("/")
}

pub fn apply_move(state: &str, mv: &str) -> Result<String> {
    let (fr, to) = decode_move(mv)?;
    let b = XiangqiBoard::parse_fen(state)?;
    let own = match b.sq[fr] {
        Some(p) if p.color == b.stm => true,
        _ => false,
    };
    if !own {
        return Err(RuntimeError::BadFen("xiangqi: bad move".to_string()));
    }
    if !pseudo_dests(&b, fr).contains(&to) {
        return Err(RuntimeError::BadFen("xiangqi: bad move".to_string()));
    }
    match b.sq[to] {
        Some(p) if p.color == b.stm => return Err(RuntimeError::BadFen("xiangqi: bad move".to_string())),
        _ => {}
    }
    let mut nb = b.clone();
    nb.sq[to] = nb.sq[fr];
    nb.sq[fr] = None;
    if king_in_check(&nb, b.stm) {
        return Err(RuntimeError::BadFen("xiangqi: bad move".to_string()));
    }
    let nstm = 1 - b.stm;
    let side = if nstm == XRED { "w" } else { "b" };
    let nmove = if b.stm == XBLACK { b.move_no + 1 } else { b.move_no };
    Ok(format!("{} {} - - 0 {}", encode_board(&nb), side, nmove))
}

pub fn game_result(state: &str) -> Result<String> {
    let b = XiangqiBoard::parse_fen(state)?;
    if legal_moves(state)?.is_empty() {
        if b.stm == XRED {
            return Ok("0-1".to_string());
        }
        return Ok("1-0".to_string());
    }
    Ok("*".to_string())
}
