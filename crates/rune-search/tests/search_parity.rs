use rune_search::lazy::LazyConfig;
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
