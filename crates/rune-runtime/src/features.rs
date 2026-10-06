use crate::board::{self, Board, BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};
use rune_spec as spec;
pub const CONTEXT_DIM: usize = 8;
pub fn vocab_size(group: usize) -> usize {
    spec::VOCAB_SIZES[group]
}
pub fn type_index(kind: u8) -> usize {
    match kind {
        PAWN => 0,
        KNIGHT => 1,
        BISHOP => 2,
        ROOK => 3,
        QUEEN => 4,
        _ => 5,
    }
}
fn sliding_attacks(b: &Board, frm: usize, target: usize) -> bool {
    let p = match b.sq[frm] {
        Some(p) => p,
        None => return false,
    };
    let df = board::sq_file(target) - board::sq_file(frm);
    let dr = board::sq_rank(target) - board::sq_rank(frm);
    let (adf, adr) = (df.abs(), dr.abs());
    let diag = p.kind == BISHOP || p.kind == QUEEN;
    let straight = p.kind == ROOK || p.kind == QUEEN;
    if !diag && !straight {
        return false;
    }
    let is_diag = adf == adr;
    let is_straight = df == 0 || dr == 0;
    if is_diag && !diag {
        return false;
    }
    if is_straight && !straight {
        return false;
    }
    if !is_diag && !is_straight {
        return false;
    }
    let sf = if df == 0 { 0 } else if df > 0 { 1 } else { -1 };
    let sr = if dr == 0 { 0 } else if dr > 0 { 1 } else { -1 };
    let (mut f, mut r) = (board::sq_file(frm) + sf, board::sq_rank(frm) + sr);
    while (f, r) != (board::sq_file(target), board::sq_rank(target)) {
        if b.sq[board::make_sq(f, r)].is_some() {
            return false;
        }
        f += sf;
        r += sr;
    }
    true
}
fn piece_attacks(b: &Board, frm: usize, target: usize) -> bool {
    let (p, color) = match b.sq[frm] {
        Some(p) => (p.kind, p.color),
        None => return false,
    };
    let df = board::sq_file(target) - board::sq_file(frm);
    let dr = board::sq_rank(target) - board::sq_rank(frm);
    let (adf, adr) = (df.abs(), dr.abs());
    if p == PAWN {
        let d = if color == WHITE { 1 } else { -1 };
        return adr == 1 && adf == 1 && dr == d;
    }
    if p == KNIGHT {
        return (adf == 1 && adr == 2) || (adf == 2 && adr == 1);
    }
    if p == KING {
        return adf <= 1 && adr <= 1;
    }
    sliding_attacks(b, frm, target)
}
pub fn pseudo_move_count(b: &Board) -> u32 {
    let color = if b.stm == 0 { WHITE } else { BLACK };
    let mut count: u32 = 0;
    for sq in 0..64 {
        let cell = match b.sq[sq] {
            Some(c) if c.color == color => c,
            _ => continue,
        };
        let (f, r) = (board::sq_file(sq), board::sq_rank(sq));
        if cell.kind == PAWN {
            let d = if color == WHITE { 1 } else { -1 };
            let promo_rank = if color == WHITE { 7 } else { 0 };
            if board::on_board(f, r + d) && b.sq[board::make_sq(f, r + d)].is_none() {
                let to = board::make_sq(f, r + d);
                count += if board::sq_rank(to) == promo_rank { 4 } else { 1 };
                let start = if color == WHITE { 1 } else { 6 };
                if r == start && b.sq[board::make_sq(f, r + 2 * d)].is_none() {
                    count += 1;
                }
            }
            for df in [-1, 1] {
                if !board::on_board(f + df, r + d) {
                    continue;
                }
                let to = board::make_sq(f + df, r + d);
                if let Some(t) = b.sq[to] {
                    if t.color != color {
                        count += if board::sq_rank(to) == promo_rank { 4 } else { 1 };
                    }
                }
                if to as i16 == b.ep_sq {
                    count += 1;
                }
            }
        } else if cell.kind == KNIGHT {
            for (df, dr) in [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)] {
                if !board::on_board(f + df, r + dr) {
                    continue;
                }
                let to = board::make_sq(f + df, r + dr);
                match b.sq[to] {
                    None => count += 1,
                    Some(t) if t.color != color => count += 1,
                    _ => {}
                }
            }
        } else if cell.kind == KING {
            for df in -1..=1 {
                for dr in -1..=1 {
                    if df == 0 && dr == 0 {
                        continue;
                    }
                    if !board::on_board(f + df, r + dr) {
                        continue;
                    }
                    let to = board::make_sq(f + df, r + dr);
                    match b.sq[to] {
                        None => count += 1,
                        Some(t) if t.color != color => count += 1,
                        _ => {}
                    }
                }
            }
            if color == WHITE && sq == board::make_sq(4, 0) {
                if (b.castle_mask & 1) != 0 && b.sq[board::make_sq(5, 0)].is_none() && b.sq[board::make_sq(6, 0)].is_none() {
                    count += 1;
                }
                if (b.castle_mask & 2) != 0 && b.sq[board::make_sq(3, 0)].is_none() && b.sq[board::make_sq(2, 0)].is_none() && b.sq[board::make_sq(1, 0)].is_none() {
                    count += 1;
                }
            }
            if color == BLACK && sq == board::make_sq(4, 7) {
                if (b.castle_mask & 4) != 0 && b.sq[board::make_sq(5, 7)].is_none() && b.sq[board::make_sq(6, 7)].is_none() {
                    count += 1;
                }
                if (b.castle_mask & 8) != 0 && b.sq[board::make_sq(3, 7)].is_none() && b.sq[board::make_sq(2, 7)].is_none() && b.sq[board::make_sq(1, 7)].is_none() {
                    count += 1;
                }
            }
        } else {
            let diag = cell.kind == BISHOP || cell.kind == QUEEN;
            let straight = cell.kind == ROOK || cell.kind == QUEEN;
            for df in -1..=1 {
                for dr in -1..=1 {
                    if df == 0 && dr == 0 {
                        continue;
                    }
                    let is_diag = df != 0 && dr != 0;
                    if is_diag && !diag {
                        continue;
                    }
                    if !is_diag && !straight {
                        continue;
                    }
                    let (mut ff, mut rr) = (f + df, r + dr);
                    while board::on_board(ff, rr) {
                        let to = board::make_sq(ff, rr);
                        match b.sq[to] {
                            None => count += 1,
                            Some(t) => {
                                if t.color != color {
                                    count += 1;
                                }
                                break;
                            }
                        }
                        ff += df;
                        rr += dr;
                    }
                }
            }
        }
    }
    count
}
pub fn extract_features(board: &Board) -> Vec<(u8, u16)> {
    let mut feats: Vec<(u8, u16)> = Vec::new();
    for sq in 0..64 {
        let cell = match board.sq[sq] {
            Some(c) => c,
            None => continue,
        };
        let ci = if cell.color == WHITE { 0 } else { 1 };
        if cell.kind == PAWN {
            feats.push((0, (ci * 64 + sq) as u16));
            feats.push((0, (128 + board::sq_file(sq) * 4 + board::sq_rank(sq) / 2) as u16));
        }
        if cell.kind == KING {
            feats.push((1, (ci * 64 + sq) as u16));
            for df in -1..=1 {
                for dr in -1..=1 {
                    if df == 0 && dr == 0 {
                        continue;
                    }
                    let (f, r) = (board::sq_file(sq) + df, board::sq_rank(sq) + dr);
                    if !board::on_board(f, r) {
                        continue;
                    }
                    let nsq = board::make_sq(f, r);
                    feats.push((1, (128 + ci * 32 + nsq / 2) as u16));
                }
            }
        }
        if cell.kind == KNIGHT {
            feats.push((2, (ci * 64 + sq) as u16));
        }
        if cell.kind == BISHOP {
            feats.push((2, (128 + ci * 64 + sq) as u16));
        }
        if cell.kind == ROOK {
            feats.push((3, (ci * 64 + sq) as u16));
        }
        if cell.kind == QUEEN {
            feats.push((4, (ci * 64 + sq) as u16));
        }
        feats.push((6, (type_index(cell.kind) * 64 + sq) as u16));
    }
    for vsq in 0..64 {
        let victim = match board.sq[vsq] {
            Some(c) => c,
            None => continue,
        };
        for asq in 0..64 {
            let attacker = match board.sq[asq] {
                Some(c) => c,
                None => continue,
            };
            if attacker.color == victim.color {
                continue;
            }
            if !piece_attacks(board, asq, vsq) {
                continue;
            }
            let vt = type_index(victim.kind);
            feats.push((5, (vt * 64 + vsq) as u16));
            let coarse = 384 + board::sq_file(asq) * 8 + board::sq_rank(asq);
            if coarse < 512 {
                feats.push((5, coarse as u16));
            }
        }
    }
    let mob = pseudo_move_count(board).min(31);
    feats.push((6, (384 + board.stm as i32 * 32 + mob as i32) as u16));
    feats.push((7, board.stm as u16));
    feats.push((7, (2 + board.castle_mask) as u16));
    let total = board.piece_count();
    let mut bucket = (total as i32 - 2) / 2;
    if bucket < 0 {
        bucket = 0;
    }
    if bucket > 15 {
        bucket = 15;
    }
    feats.push((7, (18 + bucket) as u16));
    feats.push((7, (34 + board.game_phase() as i32) as u16));
    feats.sort_unstable();
    feats.dedup();
    feats
}
pub fn diff_features(before: &[(u8, u16)], after: &[(u8, u16)]) -> (Vec<(u8, u16)>, Vec<(u8, u16)>) {
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < before.len() && j < after.len() {
        if before[i] == after[j] {
            i += 1;
            j += 1;
        } else if after[j] < before[i] {
            added.push(after[j]);
            j += 1;
        } else {
            removed.push(before[i]);
            i += 1;
        }
    }
    while j < after.len() {
        added.push(after[j]);
        j += 1;
    }
    while i < before.len() {
        removed.push(before[i]);
        i += 1;
    }
    (added, removed)
}
pub fn compute_context(board: &Board) -> Vec<f32> {
    let us = if board.stm == 0 { WHITE } else { BLACK };
    let mut pawns = 0;
    let mut minors = 0;
    let mut rooks = 0;
    let mut queens = 0;
    let mut total = 0;
    let mut king = 64;
    for sq in 0..64 {
        let cell = match board.sq[sq] {
            Some(c) => c,
            None => continue,
        };
        total += 1;
        if cell.kind == PAWN {
            pawns += 1;
        }
        if cell.kind == KNIGHT || cell.kind == BISHOP {
            minors += 1;
        }
        if cell.kind == ROOK {
            rooks += 1;
        }
        if cell.kind == QUEEN {
            queens += 1;
        }
        if cell.kind == KING && cell.color == us {
            king = sq;
        }
    }
    let mut shield = 0;
    if king < 64 {
        for df in -1..=1 {
            for dr in -1..=1 {
                if df == 0 && dr == 0 {
                    continue;
                }
                let f = board::sq_file(king) + df;
                let r = board::sq_rank(king) + dr;
                if !board::on_board(f, r) {
                    continue;
                }
                match board.sq[board::make_sq(f, r)] {
                    Some(c) if c.color == us && c.kind != KING => shield += 1,
                    _ => {}
                }
            }
        }
    }
    vec![
        board.stm as f32,
        board.game_phase() as f32 / 2.0,
        pawns as f32 / 16.0,
        minors as f32 / 8.0,
        rooks as f32 / 4.0,
        queens as f32 / 2.0,
        shield as f32 / 8.0,
        total as f32 / 32.0,
    ]
}
