// RUNE — Relational Unified Neural Evaluator
// Copyright (C) 2026 a123443212
//
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// This project is dual-licensed under the MIT License and the
// Apache License, Version 2.0. You may choose either license
// when using, copying, modifying, or distributing this software.
//
// MIT License: https://opensource.org/license/mit
// Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, this
// software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
// OR CONDITIONS OF ANY KIND, either express or implied.

use shakmaty::{Chess, Position};
use shakmaty::fen::Fen;
use std::str::FromStr;
use crate::lazy::LazyConfig;

pub const MATE_SCORE: f32 = 10000.0;

fn role_value(r: shakmaty::Role) -> i32 {
    match r {
        shakmaty::Role::Pawn => 100,
        shakmaty::Role::Knight => 320,
        shakmaty::Role::Bishop => 330,
        shakmaty::Role::Rook => 500,
        shakmaty::Role::Queen => 900,
        shakmaty::Role::King => 0,
    }
}

fn move_score(m: &shakmaty::Move) -> i32 {
    let mut score = 0;
    if let Some(v) = m.capture() {
        score = role_value(v) * 16 - role_value(m.role());
    }
    if let shakmaty::Move::Normal { promotion: Some(p), .. } = m {
        score += role_value(*p);
    }
    score
}

pub fn order_moves(moves: &mut [shakmaty::Move]) {
    moves.sort_by(|a, b| move_score(b).cmp(&move_score(a)).then_with(|| a.to_string().cmp(&b.to_string())));
}

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
    pub refined: usize,
    pub qnodes: usize,
    pub tt_hits: usize,
}

const TT_CAP: usize = 1 << 20;
const MAX_QPLY: usize = 64;

#[derive(Debug, Clone, Copy)]
struct TtEntry {
    score: f32,
    depth: usize,
    flag: u8,
}

#[derive(Debug, Default)]
struct SearchTt {
    map: std::collections::HashMap<u64, TtEntry>,
}

impl SearchTt {
    fn probe(&mut self, key: u64, depth: usize, alpha: &mut f32, beta: &mut f32, stats: &mut SearchStats) -> Option<f32> {
        let e = *self.map.get(&key)?;
        if e.depth < depth {
            return None;
        }
        stats.tt_hits += 1;
        if e.flag == 0 {
            return Some(e.score);
        }
        if e.flag == 1 && e.score > *alpha {
            *alpha = e.score;
        }
        if e.flag == 2 && e.score < *beta {
            *beta = e.score;
        }
        if *alpha >= *beta {
            return Some(e.score);
        }
        None
    }

    fn store(&mut self, key: u64, depth: usize, score: f32, alpha: f32, beta: f32) {
        if self.map.len() >= TT_CAP {
            self.map.clear();
        }
        let flag = if score <= alpha {
            2
        } else if score >= beta {
            1
        } else {
            0
        };
        if let Some(old) = self.map.get(&key) {
            if old.depth > depth {
                return;
            }
        }
        self.map.insert(key, TtEntry { score, depth, flag });
    }
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

fn parse_pos(fen: &str) -> Result<Chess, String> {
    Fen::from_str(fen)
        .map_err(|_| "bad fen".to_string())
        .and_then(|f: Fen| f.into_position(shakmaty::CastlingMode::Standard).map_err(|_| "bad position".to_string()))
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

    pub fn try_search(&mut self, fen: &str, depth: usize) -> Result<(f32, Option<String>), String> {
        let pos = parse_pos(fen)?;
        Ok(self.search_pos(&pos, depth))
    }

    pub fn try_search_lazy(&mut self, fen: &str, depth: usize, cheap: &mut impl FnMut(&str) -> f32) -> Result<(f32, Option<String>), String> {
        let pos = parse_pos(fen)?;
        Ok(self.search_lazy_pos(&pos, depth, cheap))
    }

    fn search_pos(&mut self, pos: &Chess, depth: usize) -> (f32, Option<String>) {
        let t0 = std::time::Instant::now();
        self.stats = SearchStats::default();
        let mut legal = pos.legal_moves();
        order_moves(&mut legal);
        let mut best: Option<f32> = None;
        let mut best_m = None;
        let mut alpha = f32::NEG_INFINITY;
        let beta = f32::INFINITY;
        let mut hist = vec![pos.zobrist_hash::<shakmaty::zobrist::Zobrist64>(shakmaty::EnPassantMode::Legal).0];
        let mut tt = SearchTt::default();
        for m in &legal {
            let child = apply_uci_move(pos, m);
            let v = -self.negamax(&child, depth.saturating_sub(1), 1, -beta, -alpha, &mut hist, &mut tt);
            self.stats.nodes += 1;
            self.stats.root_count += 1;
            if best.is_none_or(|b| v > b) {
                best = Some(v);
                best_m = Some(m.to_string());
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
            terminal_score(pos, 0)
        });
        self.stats.seconds = t0.elapsed().as_secs_f64();
        self.stats.root_score = s;
        self.stats.root_move = best_m.clone();
        (s, best_m)
    }

    pub fn search(&mut self, fen: &str, depth: usize) -> (f32, Option<String>) {
        match self.try_search(fen, depth) {
            Ok(v) => v,
            Err(_) => (f32::NAN, None),
        }
    }

    fn search_lazy_pos(&mut self, pos: &Chess, depth: usize, cheap: &mut impl FnMut(&str) -> f32) -> (f32, Option<String>) {
        let t0 = std::time::Instant::now();
        self.stats = SearchStats::default();
        let mut legal = pos.legal_moves();
        order_moves(&mut legal);
        let mut best: Option<f32> = None;
        let mut best_m = None;
        let mut alpha = f32::NEG_INFINITY;
        let beta = f32::INFINITY;
        let mut hist = vec![pos.zobrist_hash::<shakmaty::zobrist::Zobrist64>(shakmaty::EnPassantMode::Legal).0];
        let mut tt = SearchTt::default();
        for m in &legal {
            let child = apply_uci_move(pos, m);
            let v = -self.negamax_lazy(&child, depth.saturating_sub(1), 1, -beta, -alpha, &mut hist, &mut tt, cheap);
            self.stats.nodes += 1;
            self.stats.root_count += 1;
            if best.is_none_or(|b| v > b) {
                best = Some(v);
                best_m = Some(m.to_string());
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
            terminal_score(pos, 0)
        });
        self.stats.seconds = t0.elapsed().as_secs_f64();
        self.stats.root_score = s;
        self.stats.root_move = best_m.clone();
        (s, best_m)
    }

    pub fn search_lazy(&mut self, fen: &str, depth: usize, cheap: &mut impl FnMut(&str) -> f32) -> (f32, Option<String>) {
        match self.try_search_lazy(fen, depth, cheap) {
            Ok(v) => v,
            Err(_) => (f32::NAN, None),
        }
    }

    fn leaf_full(&mut self, pos: &Chess) -> f32 {
        self.stats.evals += 1;
        let fen = Fen::from_position(pos, shakmaty::EnPassantMode::Legal).to_string();
        (self.eval)(&fen)
    }

    fn leaf_lazy(&mut self, pos: &Chess, alpha: f32, beta: f32, cheap: &mut impl FnMut(&str) -> f32) -> f32 {
        let fen = Fen::from_position(pos, shakmaty::EnPassantMode::Legal).to_string();
        let c = cheap(&fen);
        if crate::lazy::should_refine(c, alpha, beta, 0.0, self.cfg.threshold, &self.cfg) {
            self.stats.refined += 1;
            self.stats.evals += 1;
            return (self.eval)(&fen);
        }
        c
    }

    fn negamax(&mut self, pos: &Chess, depth: usize, ply: usize, mut alpha: f32, beta: f32, hist: &mut Vec<u64>, tt: &mut SearchTt) -> f32 {
        let key: u64 = pos.zobrist_hash::<shakmaty::zobrist::Zobrist64>(shakmaty::EnPassantMode::Legal).0;
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
        let mut beta = beta;
        let orig = (alpha, beta);
        if let Some(s) = tt.probe(key, depth, &mut alpha, &mut beta, &mut self.stats) {
            hist.pop();
            self.stats.nodes += 1;
            return s;
        }
        let out = self.negamax_inner(pos, depth, ply, alpha, beta, hist, tt);
        tt.store(key, depth, out, orig.0, orig.1);
        hist.pop();
        out
    }

    fn negamax_lazy(&mut self, pos: &Chess, depth: usize, ply: usize, mut alpha: f32, beta: f32, hist: &mut Vec<u64>, tt: &mut SearchTt, cheap: &mut impl FnMut(&str) -> f32) -> f32 {
        let key: u64 = pos.zobrist_hash::<shakmaty::zobrist::Zobrist64>(shakmaty::EnPassantMode::Legal).0;
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
        let mut beta = beta;
        let orig = (alpha, beta);
        if let Some(s) = tt.probe(key, depth, &mut alpha, &mut beta, &mut self.stats) {
            hist.pop();
            self.stats.nodes += 1;
            return s;
        }
        let out = self.negamax_lazy_inner(pos, depth, ply, alpha, beta, hist, tt, cheap);
        tt.store(key, depth, out, orig.0, orig.1);
        hist.pop();
        out
    }

    fn negamax_inner(&mut self, pos: &Chess, depth: usize, ply: usize, mut alpha: f32, beta: f32, hist: &mut Vec<u64>, tt: &mut SearchTt) -> f32 {
        if depth == 0 {
            let stand = self.leaf_full(pos);
            return self.qsearch(pos, stand, alpha, beta, ply, 0, hist, tt);
        }
        let mut moves = pos.legal_moves();
        order_moves(&mut moves);
        if moves.is_empty() {
            self.stats.nodes += 1;
            self.stats.evals += 1;
            self.stats.leaf_count += 1;
            return terminal_score(pos, ply);
        }
        let mut best = f32::NEG_INFINITY;
        for m in &moves {
            let child = apply_uci_move(pos, m);
            let v = -self.negamax(&child, depth - 1, ply + 1, -beta, -alpha, hist, tt);
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
        best
    }

    fn negamax_lazy_inner(&mut self, pos: &Chess, depth: usize, ply: usize, mut alpha: f32, beta: f32, hist: &mut Vec<u64>, tt: &mut SearchTt, cheap: &mut impl FnMut(&str) -> f32) -> f32 {
        if depth == 0 {
            let stand = self.leaf_lazy(pos, alpha, beta, cheap);
            return self.qsearch(pos, stand, alpha, beta, ply, 0, hist, tt);
        }
        let mut moves = pos.legal_moves();
        order_moves(&mut moves);
        if moves.is_empty() {
            self.stats.nodes += 1;
            self.stats.evals += 1;
            self.stats.leaf_count += 1;
            return terminal_score(pos, ply);
        }
        let mut best = f32::NEG_INFINITY;
        for m in &moves {
            let child = apply_uci_move(pos, m);
            let v = -self.negamax_lazy(&child, depth - 1, ply + 1, -beta, -alpha, hist, tt, cheap);
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
        best
    }

    fn qsearch(&mut self, pos: &Chess, stand: f32, mut alpha: f32, beta: f32, ply: usize, qply: usize, hist: &mut Vec<u64>, tt: &mut SearchTt) -> f32 {
        self.stats.nodes += 1;
        self.stats.leaf_count += 1;
        self.stats.qnodes += 1;
        if qply >= MAX_QPLY {
            return stand;
        }
        let key: u64 = pos.zobrist_hash::<shakmaty::zobrist::Zobrist64>(shakmaty::EnPassantMode::Legal).0;
        hist.push(key);
        if hist.iter().filter(|h| **h == key).count() >= 3 {
            hist.pop();
            return 0.0;
        }
        let mut caps: Vec<shakmaty::Move> = pos.legal_moves().into_iter().filter(|m| m.is_capture()).collect();
        order_moves(&mut caps);
        if caps.is_empty() {
            hist.pop();
            return stand;
        }
        let mut best = stand;
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            hist.pop();
            return best;
        }
        for m in &caps {
            let child = apply_uci_move(pos, m);
            let child_stand = self.leaf_full(&child);
            let v = -self.qsearch(&child, child_stand, -beta, -alpha, ply + 1, qply + 1, hist, tt);
            self.stats.nodes += 1;
            self.stats.leaf_count += 1;
            self.stats.qnodes += 1;
            if v > best {
                best = v;
            }
            if v > alpha {
                alpha = v;
            }
            if alpha >= beta {
                break;
            }
        }
        hist.pop();
        best
    }
}

fn apply_uci_move(pos: &Chess, m: &shakmaty::Move) -> Chess {
    pos.clone().play(m.clone()).unwrap_or_else(|_| pos.clone())
}
