use crate::error::{Error, Result};
use crate::hash;

pub const EMPTY: u8 = 0;
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
pub struct Board {
    pub sq: [Option<Piece>; 64],
    pub stm: u8,
    pub castle_mask: u8,
    pub ep_sq: i16,
    pub halfmove_clock: u16,
    pub fullmove_number: u16,
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
    (0..8).contains(&f) && (0..8).contains(&r)
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

impl Board {
    pub fn parse_fen(fen: &str) -> Result<Board> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(Error::BadFen("expected >= 4 fields".to_string()));
        }
        let mut sq: [Option<Piece>; 64] = [None; 64];
        let mut rank: i32 = 7;
        let mut file: i32 = 0;
        for c in parts[0].chars() {
            if c == '/' {
                if file != 8 {
                    return Err(Error::BadFen("short rank".to_string()));
                }
                rank -= 1;
                file = 0;
                if rank < 0 {
                    return Err(Error::BadFen("too many ranks".to_string()));
                }
            } else if c.is_ascii_digit() {
                file += c.to_digit(10).unwrap() as i32;
                if file > 8 {
                    return Err(Error::BadFen("rank overflow".to_string()));
                }
            } else if let Some((kind, color)) = parse_piece(c) {
                if file >= 8 || rank < 0 {
                    return Err(Error::BadFen("square overflow".to_string()));
                }
                sq[make_sq(file, rank)] = Some(Piece { kind, color });
                file += 1;
            } else {
                return Err(Error::BadFen(format!("bad char {c}")));
            }
        }
        if rank != 0 || file != 8 {
            return Err(Error::BadFen("bad placement".to_string()));
        }
        let stm = match parts[1] {
            "w" => WHITE,
            "b" => BLACK,
            _ => return Err(Error::BadFen("bad side".to_string())),
        };
        let mut mask: u8 = 0;
        if parts[2] != "-" {
            for c in parts[2].chars() {
                match c {
                    'K' => mask |= 1,
                    'Q' => mask |= 2,
                    'k' => mask |= 4,
                    'q' => mask |= 8,
                    _ => return Err(Error::BadFen("bad castling".to_string())),
                }
            }
        }
        let ep_sq: i16 = if parts[3] == "-" {
            -1
        } else {
            let b = parts[3].as_bytes();
            if b.len() != 2 || !(b'a'..=b'h').contains(&b[0]) || !(b'1'..=b'8').contains(&b[1]) {
                return Err(Error::BadFen("bad ep".to_string()));
            }
            make_sq((b[0] - b'a') as i32, (b[1] - b'1') as i32) as i16
        };
        Ok(Board {
            sq,
            stm,
            castle_mask: mask,
            ep_sq,
            halfmove_clock: parts
                .get(4)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            fullmove_number: parts
                .get(5)
                .and_then(|s| s.parse().ok())
                .filter(|n| *n != 0)
                .unwrap_or(1),
        })
    }

    pub fn normalized_key(fen: &str) -> String {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        parts[..parts.len().min(4)].join(" ")
    }

    pub fn piece_count(&self) -> usize {
        self.sq.iter().filter(|c| c.is_some()).count()
    }

    pub fn non_pawn_king_count(&self) -> usize {
        self.sq
            .iter()
            .filter(|c| matches!(c, Some(p) if p.kind != PAWN && p.kind != KING))
            .count()
    }

    pub fn game_phase(&self) -> u8 {
        let n = self.non_pawn_king_count();
        if n >= 12 {
            0
        } else if n >= 6 {
            1
        } else {
            2
        }
    }

    pub fn material(&self, color: u8) -> u32 {
        self.sq
            .iter()
            .filter_map(|c| *c)
            .filter(|p| p.color == color)
            .map(|p| match p.kind {
                PAWN => 1,
                KNIGHT | BISHOP => 3,
                ROOK => 5,
                QUEEN => 9,
                _ => 0,
            })
            .sum()
    }

    pub fn queen_count(&self) -> u8 {
        self.sq
            .iter()
            .filter(|c| matches!(c, Some(p) if p.kind == QUEEN))
            .count() as u8
    }

    pub fn pawn_count(&self) -> u8 {
        self.sq
            .iter()
            .filter(|c| matches!(c, Some(p) if p.kind == PAWN))
            .count() as u8
    }

    pub fn canonical_strict(fen: &str) -> String {
        fen.split_whitespace().take(4).collect::<Vec<_>>().join(" ")
    }
}

pub fn canonical_identity(fen: &str, strict: bool) -> u64 {
    if strict {
        hash::fnv1a_str(&Board::canonical_strict(fen))
    } else {
        hash::fnv1a_str(&Board::normalized_key(fen))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_startpos() {
        let b =
            Board::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert_eq!(b.piece_count(), 32);
        assert_eq!(b.stm, WHITE);
        assert_eq!(b.castle_mask, 15);
        assert_eq!(b.ep_sq, -1);
        assert_eq!(b.game_phase(), 0);
    }

    #[test]
    fn parse_rejects_bad() {
        assert!(Board::parse_fen("nope").is_err());
        assert!(Board::parse_fen("8/8/8/8/8/8/8/8 w - - 0 1").is_ok());
    }

    #[test]
    fn phase_boundaries() {
        let end = Board::parse_fen("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1").unwrap();
        assert_eq!(end.game_phase(), 2);
    }
}
