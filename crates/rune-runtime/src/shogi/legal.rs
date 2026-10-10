use crate::error::{Result, RuntimeError};
use super::moves::{hand_index, pseudo_moves, unpromote, SMove};
use super::{piece_attacks, ShogiBoard, B, DR, G, HB, K, L, N, P, PL, PN, PP, PS, R, S, SBLACK, SWHITE};

const HAND_LETTERS: [char; 7] = ['p', 'l', 'n', 's', 'g', 'b', 'r'];

fn kind_letter(kind: u8) -> Option<char> {
    match hand_index(kind) {
        Some(i) => Some(HAND_LETTERS[i]),
        None => None,
    }
}

fn letter_kind(c: char) -> Option<u8> {
    match c {
        'p' => Some(P),
        'l' => Some(L),
        'n' => Some(N),
        's' => Some(S),
        'g' => Some(G),
        'b' => Some(B),
        'r' => Some(R),
        _ => None,
    }
}

pub fn encode_move(mv: SMove) -> String {
    match mv {
        SMove::Normal { fr, to, promo } => {
            if promo {
                format!("{:02}{:02}+", fr, to)
            } else {
                format!("{:02}{:02}", fr, to)
            }
        }
        SMove::Drop { to, piece } => {
            format!("D{}{}", to, kind_letter(piece).unwrap_or('p'))
        }
    }
}

pub fn decode_move(mv: &str) -> Result<SMove> {
    if let Some(body) = mv.strip_prefix('D') {
        let mut to_end = body.len();
        for (i, c) in body.char_indices() {
            if c.is_ascii_alphabetic() {
                to_end = i;
                break;
            }
        }
        let piece = letter_kind(body[to_end..].chars().next().ok_or_else(|| RuntimeError::BadFen("shogi: bad move".to_string()))?)
            .ok_or_else(|| RuntimeError::BadFen("shogi: bad move".to_string()))?;
        let to: usize = body[..to_end].parse().map_err(|_| RuntimeError::BadFen("shogi: bad move".to_string()))?;
        if to >= 81 {
            return Err(RuntimeError::BadFen("shogi: bad move".to_string()));
        }
        return Ok(SMove::Drop { to, piece });
    }
    let promo = mv.ends_with('+');
    let core = if promo { &mv[..mv.len() - 1] } else { mv };
    if core.len() != 4 {
        return Err(RuntimeError::BadFen("shogi: bad move".to_string()));
    }
    let fr: usize = core[0..2].parse().map_err(|_| RuntimeError::BadFen("shogi: bad move".to_string()))?;
    let to: usize = core[2..4].parse().map_err(|_| RuntimeError::BadFen("shogi: bad move".to_string()))?;
    if fr >= 81 || to >= 81 {
        return Err(RuntimeError::BadFen("shogi: bad move".to_string()));
    }
    Ok(SMove::Normal { fr, to, promo })
}

pub fn king_in_check(b: &ShogiBoard, color: u8) -> bool {
    let mut kus: Option<usize> = None;
    for sq in 0..81 {
        match b.sq[sq] {
            Some(p) if p.kind == K && p.color == color => {
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
    for asq in 0..81 {
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

fn promote_kind(kind: u8) -> Option<u8> {
    match kind {
        P => Some(PP),
        L => Some(PL),
        N => Some(PN),
        S => Some(PS),
        B => Some(HB),
        R => Some(DR),
        _ => None,
    }
}

fn do_move(b: &ShogiBoard, mv: SMove) -> Result<ShogiBoard> {
    let mut nb = b.clone();
    match mv {
        SMove::Drop { to, piece } => {
            let ti = hand_index(piece).ok_or_else(|| RuntimeError::BadFen("shogi: bad move".to_string()))?;
            if nb.hand[b.stm as usize][ti] == 0 {
                return Err(RuntimeError::BadFen("shogi: bad move".to_string()));
            }
            if nb.sq[to].is_some() {
                return Err(RuntimeError::BadFen("shogi: bad move".to_string()));
            }
            nb.hand[b.stm as usize][ti] -= 1;
            nb.sq[to] = Some(super::SPiece { kind: piece, color: b.stm });
        }
        SMove::Normal { fr, to, promo } => {
            let (kind, color) = match nb.sq[fr] {
                Some(p) => (p.kind, p.color),
                None => return Err(RuntimeError::BadFen("shogi: bad move".to_string())),
            };
            if color != b.stm {
                return Err(RuntimeError::BadFen("shogi: bad move".to_string()));
            }
            let nk = if promo {
                promote_kind(kind).ok_or_else(|| RuntimeError::BadFen("shogi: bad move".to_string()))?
            } else {
                kind
            };
            match nb.sq[to] {
                Some(p) if p.color == b.stm => return Err(RuntimeError::BadFen("shogi: bad move".to_string())),
                Some(p) => {
                    let base = unpromote(p.kind);
                    let ti = hand_index(base).ok_or_else(|| RuntimeError::BadFen("shogi: bad move".to_string()))?;
                    nb.hand[b.stm as usize][ti] = nb.hand[b.stm as usize][ti].saturating_add(1);
                }
                None => {}
            }
            nb.sq[to] = Some(super::SPiece { kind: nk, color: b.stm });
            nb.sq[fr] = None;
        }
    }
    Ok(nb)
}

fn serialize_board(b: &ShogiBoard) -> String {
    let mut ranks = Vec::new();
    for ri in 0..9 {
        let mut rank = String::new();
        let mut gap = 0;
        for f in 0..9 {
            match b.sq[ri * 9 + f] {
                None => gap += 1,
                Some(p) => {
                    if gap > 0 {
                        rank.push_str(&gap.to_string());
                        gap = 0;
                    }
                    let base = unpromote(p.kind);
                    let promoted = base != p.kind;
                    let ch = match base {
                        P => 'p',
                        L => 'l',
                        N => 'n',
                        S => 's',
                        G => 'g',
                        B => 'b',
                        R => 'r',
                        K => 'k',
                        _ => 'p',
                    };
                    if promoted {
                        rank.push('+');
                    }
                    if p.color == SBLACK {
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
        ranks.push(rank);
    }
    ranks.join("/")
}

fn serialize_hands(b: &ShogiBoard) -> String {
    let mut s = String::new();
    for color in [SBLACK, SWHITE] {
        for (ti, t) in ['r', 'b', 'g', 's', 'n', 'l', 'p'].iter().enumerate() {
            let n = b.hand[color as usize][ti];
            if n == 0 {
                continue;
            }
            if n > 1 {
                s.push_str(&n.to_string());
            }
            if color == SBLACK {
                s.push(t.to_ascii_uppercase());
            } else {
                s.push(*t);
            }
        }
    }
    if s.is_empty() {
        s.push('-');
    }
    s
}

fn pawn_drop_mate(b: &ShogiBoard, to: usize) -> bool {
    let mut nb = b.clone();
    nb.sq[to] = Some(super::SPiece { kind: P, color: b.stm });
    let foe = 1 - b.stm;
    if !king_in_check(&nb, foe) {
        return false;
    }
    for frm in 0..81 {
        let own = match nb.sq[frm] {
            Some(p) if p.color == foe => true,
            _ => false,
        };
        if !own {
            continue;
        }
        let nb_foe = nb_for(&nb, foe);
        for mv in super::moves::pseudo_moves(&nb_foe) {
            match mv {
                SMove::Drop { .. } => {}
                SMove::Normal { fr, to: t, promo } => {
                    if fr != frm {
                        continue;
                    }
                    if t == to {
                        return false;
                    }
                    let mut trial = nb.clone();
                    let kind = trial.sq[fr].map(|p| p.kind).unwrap_or(P);
                    let nk = if promo {
                        promote_kind(kind).unwrap_or(kind)
                    } else {
                        kind
                    };
                    trial.sq[t] = Some(super::SPiece { kind: nk, color: foe });
                    trial.sq[fr] = None;
                    if !king_in_check(&trial, foe) {
                        return false;
                    }
                }
            }
        }
    }
    true
}

fn nb_for(b: &ShogiBoard, stm: u8) -> ShogiBoard {
    let mut nb = b.clone();
    nb.stm = stm;
    nb
}

pub fn legal_moves(state: &str) -> Result<Vec<String>> {
    let b = ShogiBoard::parse_sfen(state)?;
    let mut out = Vec::new();
    for mv in pseudo_moves(&b) {
        match mv {
            SMove::Drop { to, piece } => {
                if b.sq[to].is_some() {
                    continue;
                }
                if piece == P && pawn_drop_mate(&b, to) {
                    continue;
                }
                let mut nb = b.clone();
                let ti = hand_index(piece).unwrap_or(0);
                nb.hand[b.stm as usize][ti] -= 1;
                nb.sq[to] = Some(super::SPiece { kind: piece, color: b.stm });
                if king_in_check(&nb, b.stm) {
                    continue;
                }
                out.push(encode_move(mv));
            }
            SMove::Normal { .. } => {
                let nb = do_move(&b, mv)?;
                if king_in_check(&nb, b.stm) {
                    continue;
                }
                out.push(encode_move(mv));
            }
        }
    }
    out.sort();
    Ok(out)
}

pub fn apply_move(state: &str, mv: &str) -> Result<String> {
    let b = ShogiBoard::parse_sfen(state)?;
    let m = decode_move(mv)?;
    if !pseudo_moves(&b).contains(&m) {
        return Err(RuntimeError::BadFen("shogi: bad move".to_string()));
    }
    match m {
        SMove::Drop { to, piece } => {
            if b.sq[to].is_some() {
                return Err(RuntimeError::BadFen("shogi: bad move".to_string()));
            }
            if piece == P && pawn_drop_mate(&b, to) {
                return Err(RuntimeError::BadFen("shogi: bad move".to_string()));
            }
        }
        SMove::Normal { .. } => {}
    }
    let mut nb = do_move(&b, m)?;
    if king_in_check(&nb, b.stm) {
        return Err(RuntimeError::BadFen("shogi: bad move".to_string()));
    }
    nb.stm = 1 - b.stm;
    nb.move_no = b.move_no + 1;
    let side = if nb.stm == SBLACK { "b" } else { "w" };
    Ok(format!("{} {} {} {}", serialize_board(&nb), side, serialize_hands(&nb), nb.move_no))
}

pub fn game_result(state: &str) -> Result<String> {
    let b = ShogiBoard::parse_sfen(state)?;
    if legal_moves(state)?.is_empty() {
        if b.stm == SBLACK {
            return Ok("0-1".to_string());
        }
        return Ok("1-0".to_string());
    }
    Ok("*".to_string())
}
