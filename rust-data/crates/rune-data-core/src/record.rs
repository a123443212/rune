use crate::board::Board;
use crate::features::{extract_features, VOCAB_SIZES};
use crate::hash;

#[derive(Debug, Clone)]
pub struct Record {
    pub fen: String,
    pub identity: u64,
    pub phase: u8,
    pub stm: u8,
    pub piece_count: u8,
    pub pawn_count: u8,
    pub queens: u8,
    pub imbalance_bucket: u8,
    pub teacher_value: f32,
    pub teacher_wdl: u8,
    pub has_teacher: bool,
    pub teacher_cp: i32,
    pub perspective_stm: bool,
    pub game_hash: u64,
    pub ply: u16,
    pub source_id: u32,
    pub features: Vec<(u8, u16)>,
}

impl Record {
    pub fn from_fen(
        fen: &str,
        game_id: &str,
        ply: u16,
        source_id: u32,
        strict_identity: bool,
    ) -> Result<Self, crate::error::Error> {
        let board = Board::parse_fen(fen)?;
        let feats = extract_features(&board);
        for &(g, i) in &feats {
            debug_assert!((i as usize) < VOCAB_SIZES[g as usize]);
        }
        let w = board.material(crate::board::WHITE) as i32;
        let b = board.material(crate::board::BLACK) as i32;
        Ok(Record {
            fen: fen.to_string(),
            identity: crate::board::canonical_identity(fen, strict_identity),
            phase: board.game_phase(),
            stm: board.stm,
            piece_count: board.piece_count().min(255) as u8,
            pawn_count: board.pawn_count(),
            queens: board.queen_count(),
            imbalance_bucket: ((w - b).abs() / 3).min(7) as u8,
            teacher_value: 0.0,
            teacher_wdl: 1,
            has_teacher: false,
            teacher_cp: 0,
            perspective_stm: true,
            game_hash: hash::fnv1a_str(game_id),
            ply,
            source_id,
            features: feats,
        })
    }

    pub fn with_teacher(mut self, value: f32, wdl: u8, cp: i32, perspective_stm: bool) -> Self {
        self.teacher_value = value;
        self.teacher_wdl = wdl;
        self.has_teacher = true;
        self.teacher_cp = cp;
        self.perspective_stm = perspective_stm;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_round_fields() {
        let r = Record::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "g1",
            0,
            0,
            false,
        )
        .unwrap();
        assert_eq!(r.phase, 0);
        assert_eq!(r.piece_count, 32);
        assert!(!r.features.is_empty());
        assert!(!r.has_teacher);
    }
}
