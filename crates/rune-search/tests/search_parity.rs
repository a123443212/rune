use rune_search::lazy::{LazyConfig, LazyMode};
use rune_search::AlphaBeta;

const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[test]
fn search_deterministic() {
    let mut eval = |_: &str| 0.05f32;
    let mut a = AlphaBeta::new(&mut eval, LazyConfig::default());
    let (s1, m1) = a.search(STARTPOS, 2);
    let n1 = a.stats.nodes;
    let mut eval2 = |_: &str| 0.05f32;
    let mut b = AlphaBeta::new(&mut eval2, LazyConfig::default());
    let (s2, m2) = b.search(STARTPOS, 2);
    assert_eq!(s1, s2);
    assert_eq!(m1, m2);
    assert_eq!(n1, b.stats.nodes);
    assert!(n1 > 0);
}

#[test]
fn search_counts_node_types() {
    let mut eval = |_: &str| 0.05f32;
    let mut ab = AlphaBeta::new(&mut eval, LazyConfig::default());
    ab.search(STARTPOS, 2);
    assert!(ab.stats.root_count > 0);
    assert!(ab.stats.leaf_count > 0);
    assert_eq!(ab.stats.nodes, ab.stats.root_count + ab.stats.pv_count + ab.stats.cut_count + ab.stats.leaf_count);
}

const FOOLS_MATE: &str = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1";
const STALEMATE: &str = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1";

#[test]
fn search_scores_mate_and_stalemate() {
    let mut eval = |_: &str| 0.5f32;
    let mut a = AlphaBeta::new(&mut eval, LazyConfig::default());
    let (s, m) = a.search(FOOLS_MATE, 2);
    assert!(s < -(rune_search::search::MATE_SCORE - 10.0), "{}", s);
    assert!(m.is_none());
    let mut eval2 = |_: &str| 0.5f32;
    let mut b = AlphaBeta::new(&mut eval2, LazyConfig::default());
    let (s2, m2) = b.search(STALEMATE, 2);
    assert!((s2 - 0.0).abs() < 1e-6, "{}", s2);
    assert!(m2.is_none());
}

const FIFTY: &str = "6k1/8/8/8/8/8/8/K6R w - - 100 45";
#[test]
fn search_returns_draw_on_fifty_moves() {
    let mut eval = |_: &str| 0.9f32;
    let mut a = AlphaBeta::new(&mut eval, LazyConfig::default());
    let (s, _) = a.search(FIFTY, 2);
    assert!((s - 0.0).abs() < 1e-6, "{}", s);
}

#[test]
fn ordering_puts_captures_first() {
    use shakmaty::{Chess, Position};
    let fen: shakmaty::fen::Fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3".parse().unwrap();
    let pos: Chess = fen.into_position(shakmaty::CastlingMode::Standard).unwrap();
    let mut moves = pos.legal_moves();
    assert!(!moves.is_empty());
    rune_search::search::order_moves(&mut moves);
    assert!(moves[0].is_capture());
}

#[test]
fn lazy_skips_and_refines() {
    use std::cell::Cell;
    let full_calls = Cell::new(0usize);
    let mut full = |_: &str| {
        full_calls.set(full_calls.get() + 1);
        0.05f32
    };
    let cfg = LazyConfig { mode: LazyMode::L1, threshold: 0.5, ..Default::default() };
    let mut cheap_low = |_: &str| 0.0f32;
    let mut ab = AlphaBeta::new(&mut full, cfg);
    let (s, _) = ab.search_lazy(STARTPOS, 1, &mut cheap_low);
    assert_eq!(full_calls.get(), 0);
    assert_eq!(ab.stats.refined, 0);
    assert!((s - 0.0).abs() < 1e-6, "{}", s);
    let mut cheap_high = |_: &str| 0.9f32;
    let mut ab2 = AlphaBeta::new(&mut full, LazyConfig { mode: LazyMode::L1, threshold: 0.5, ..Default::default() });
    let _ = ab2.search_lazy(STARTPOS, 1, &mut cheap_high);
    assert!(full_calls.get() > 0);
    assert!(ab2.stats.refined > 0);
}

#[test]
fn quiescence_sees_hanging_rook() {
    use shakmaty::{Chess, Position};
    fn material(pos: &Chess) -> f32 {
        let mut s = 0;
        for sq in shakmaty::Square::ALL {
            if let Some(p) = pos.board().piece_at(sq) {
                let v = match p.role {
                    shakmaty::Role::Pawn => 100,
                    shakmaty::Role::Knight => 320,
                    shakmaty::Role::Bishop => 330,
                    shakmaty::Role::Rook => 500,
                    shakmaty::Role::Queen => 900,
                    shakmaty::Role::King => 0,
                };
                s += if p.color == pos.turn() { v } else { -v };
            }
        }
        s as f32 / 1000.0
    }
    let mut eval = |fen: &str| {
        let f: shakmaty::fen::Fen = fen.parse().unwrap();
        let pos: Chess = f.into_position(shakmaty::CastlingMode::Standard).unwrap();
        material(&pos)
    };
    let mut ab = AlphaBeta::new(&mut eval, LazyConfig::default());
    let (s, m) = ab.search("r3k3/8/8/8/8/8/8/R3K3 w - - 0 1", 0);
    assert!(m.is_some());
    assert!(s > 0.4, "{}", s);
    assert!(ab.stats.qnodes > 0);
}

#[test]
fn tt_records_hits() {
    let mut eval = |_: &str| 0.05f32;
    let mut ab = AlphaBeta::new(&mut eval, LazyConfig::default());
    ab.search(STARTPOS, 3);
    assert!(ab.stats.tt_hits > 0, "hits {}", ab.stats.tt_hits);
}
