use shakmaty::{Chess, Position};
use shakmaty::fen::Fen;
use shakmaty::zobrist::Zobrist64;
use std::str::FromStr;
use crate::lazy::LazyConfig;

pub const MATE_SCORE: f32 = 10000.0;

fn terminal_score(pos: &Chess, ply: usize) -> f32 {
    if pos.is_checkmate() {
        return -(MATE_SCORE - ply as f32);
    }
    0.0
}

#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    pub nodes: usize,
    pub evals: usize,
    pub cutoffs: usize,
    pub seconds: f64,
    pub root_score: f32,
    pub root_move: Option<String>,
    pub root_count: usize,
    pub pv_count: usize,
    pub cut_count: usize,
    pub leaf_count: usize,
}

impl SearchStats {
    pub fn nps(&self) -> f64 {
        if self.seconds > 0.0 {
            self.nodes as f64 / self.seconds
        } else {
            0.0
        }
    }
}

fn parse_pos(fen: &str) -> Chess {
    Fen::from_str(fen).ok().and_then(|f: Fen| f.into_position(shakmaty::CastlingMode::Standard).ok()).unwrap_or_default()
}

pub struct AlphaBeta<F>
where
    F: FnMut(&str) -> f32,
{
    pub eval: F,
    pub cfg: LazyConfig,
    pub stats: SearchStats,
}

impl<F> AlphaBeta<F>
where
    F: FnMut(&str) -> f32,
{
    pub fn new(eval: F, cfg: LazyConfig) -> Self {
        AlphaBeta { eval, cfg, stats: SearchStats::default() }
    }

    pub fn search(&mut self, fen: &str, depth: usize) -> (f32, Option<String>) {
        let t0 = std::time::Instant::now();
        self.stats = SearchStats::default();
        let pos = parse_pos(fen);
        let moves: Vec<String> = pos.legal_moves().iter().map(|m| m.to_string()).collect();
        let mut best: Option<f32> = None;
        let mut best_m = None;
        let mut alpha = f32::NEG_INFINITY;
        let beta = f32::INFINITY;
        let mut hist = vec![pos.zobrist_hash(shakmaty::EnPassantMode::Legal)];
        for m in &moves {
            let child = apply_uci(&pos, m);
            let v = -self.negamax(&child, depth.saturating_sub(1), 1, -beta, -alpha, &mut hist);
            self.stats.nodes += 1;
            self.stats.root_count += 1;
            if best.is_none_or(|b| v > b) {
                best = Some(v);
                best_m = Some(m.clone());
            }
            if v > alpha {
                alpha = v;
            }
            if alpha >= beta {
                self.stats.cutoffs += 1;
                break;
            }
        }
        let s = best.unwrap_or_else(|| {
            self.stats.nodes += 1;
            self.stats.evals += 1;
            terminal_score(&pos, 0)
        });
        self.stats.seconds = t0.elapsed().as_secs_f64();
        self.stats.root_score = s;
        self.stats.root_move = best_m.clone();
        (s, best_m)
    }

    fn negamax(&mut self, pos: &Chess, depth: usize, ply: usize, mut alpha: f32, beta: f32, hist: &mut Vec<Zobrist64>) -> f32 {
        let key: Zobrist64 = pos.zobrist_hash(shakmaty::EnPassantMode::Legal);
        hist.push(key);
        if hist.iter().filter(|h| **h == key).count() >= 3 {
            hist.pop();
            self.stats.nodes += 1;
            return 0.0;
        }
        if pos.halfmoves() >= 100 {
            hist.pop();
            self.stats.nodes += 1;
            return 0.0;
        }
        if depth == 0 {
            self.stats.nodes += 1;
            self.stats.evals += 1;
            self.stats.leaf_count += 1;
            let fen = Fen::from_position(pos, shakmaty::EnPassantMode::Legal).to_string();
            hist.pop();
            return (self.eval)(&fen);
        }
        let moves = pos.legal_moves();
        if moves.is_empty() {
            self.stats.nodes += 1;
            self.stats.evals += 1;
            self.stats.leaf_count += 1;
            hist.pop();
            return terminal_score(pos, ply);
        }
        let mut best = f32::NEG_INFINITY;
        for m in &moves {
            let child = apply_uci_move(pos, m);
            let v = -self.negamax(&child, depth - 1, ply + 1, -beta, -alpha, hist);
            self.stats.nodes += 1;
            if beta - alpha > 1.0 {
                self.stats.pv_count += 1;
            } else if ply % 2 == 0 {
                self.stats.cut_count += 1;
            } else {
                self.stats.leaf_count += 1;
            }
            if v > best {
                best = v;
            }
            if v > alpha {
                alpha = v;
            }
            if alpha >= beta {
                self.stats.cutoffs += 1;
                break;
            }
        }
        hist.pop();
        best
    }
}

fn apply_uci(pos: &Chess, uci: &str) -> Chess {
    use std::str::FromStr;
    match shakmaty::uci::UciMove::from_str(uci) {
        Ok(u) => match u.to_move(pos) {
            Ok(m) => pos.clone().play(m).unwrap_or_else(|_| pos.clone()),
            Err(_) => pos.clone(),
        },
        Err(_) => pos.clone(),
    }
}

fn apply_uci_move(pos: &Chess, m: &shakmaty::Move) -> Chess {
    pos.clone().play(m.clone()).unwrap_or_else(|_| pos.clone())
}
