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

use std::path::Path;
use rune_kernel as kernel;
use rune_model as model;
use rune_spec as spec;
use crate::accumulator::{Accumulator, Tables};
use crate::board::Board;
use crate::error::{Result, RuntimeError};
use crate::features::{compute_context, diff_features, extract_features};
use crate::mixer::{HeadTrace, HeadWeights, MixerTrace, MixerWeights};
use crate::mixer_dual::DualMixer;
use crate::mixer_mh::MultiHeadMixer;
use crate::model_config::{resolve_model_config, ExecutionTarget};
use crate::evaluator_spatial::SpatialExecutor;
use crate::sparse_model::SparseModel;

pub(crate) const FULL_REFRESH_LIMIT: usize = 64;
#[derive(Debug, Clone)]
pub struct EvalResult {
    pub value: f32,
    pub wdl: [f32; 3],
    pub refine: bool,
    pub difficulty: f32,
    pub policy: Vec<f32>,
    pub score_mean: f32,
}
#[derive(Debug, Clone, Default)]
pub struct FullTrace {
    pub features: Vec<(u8, u16)>,
    pub accumulator: Vec<f32>,
    pub tokens: Vec<f32>,
    pub mixer: MixerTrace,
    pub head: HeadTrace,
    pub value: f32,
    pub wdl: [f32; 3],
}
pub struct Evaluator {
    tables: Tables,
    mixer: Option<MixerWeights>,
    dual: Option<DualMixer>,
    mh: Option<MultiHeadMixer>,
    heads: Vec<HeadWeights>,
    pub(crate) spatial: Option<SpatialExecutor>,
    phase: u8,
    arch_id: String,
    game: String,
    pub(crate) tokens: usize,
    pub(crate) dim: usize,
    acc: Accumulator,
    feats: Vec<(u8, u16)>,
    ctx: Vec<f32>,
    stack: Vec<(Vec<f32>, Vec<(u8, u16)>, Vec<f32>, u8)>,
}
impl Evaluator {
    pub fn load(path: &Path) -> Result<Evaluator> {
        let m = model::load(path).map_err(|e| RuntimeError::InvalidState(e.to_string()))?;
        Evaluator::from_model(&m)
    }
    pub fn from_model(m: &model::RuneModel) -> Result<Evaluator> {
        let config = resolve_model_config(m, ExecutionTarget::Reference)?;
        let game = config.game.clone();
        let arch = config.architecture_id.clone();
        let tokens = config.tokens;
        let dim = config.dim;
        let vocabs = config.vocabs;
        let is_resnet = config.is_resnet;
        if is_resnet {
            let tables = Tables::zeros(dim, vocabs);
            let spatial = SpatialExecutor::from_model(m)?;
            let acc = Accumulator::new(tokens, dim);
            return Ok(Evaluator { tables, mixer: None, dual: None, mh: None, heads: Vec::new(), spatial: Some(spatial), phase: 1, arch_id: arch, game, tokens, dim, acc, feats: Vec::new(), ctx: Vec::new(), stack: Vec::new() });
        }
        let sparse = SparseModel::from_model(m, &config)?;
        let acc = Accumulator::new(tokens, dim);
        Ok(Evaluator { tables: sparse.tables, mixer: sparse.mixer, dual: sparse.dual, mh: sparse.multi_head, heads: sparse.heads, spatial: None, phase: 1, arch_id: arch, game, tokens, dim, acc, feats: Vec::new(), ctx: Vec::new(), stack: Vec::new() })
    }
    pub fn arch_id(&self) -> &str {
        &self.arch_id
    }
    pub fn game_id(&self) -> &str {
        &self.game
    }
    pub fn phase(&self) -> u8 {
        self.phase
    }
    pub fn head_count(&self) -> usize {
        self.heads.len()
    }
    pub fn ctx_of(&self) -> &[f32] {
        &self.ctx
    }
    fn active_ctx(&self) -> Option<&[f32]> {
        if self.ctx.is_empty() {
            return None;
        }
        Some(&self.ctx)
    }
    pub fn refresh(&mut self, board: &Board) -> Result<()> {
        if self.game != spec::GAME_CHESS {
            return Err(RuntimeError::InvalidState(format!("chess refresh on {} model", self.game)));
        }
        let f = extract_features(board);
        self.acc.refresh(&self.tables, &f);
        self.feats = f;
        self.ctx = compute_context(board);
        self.phase = board.game_phase();
        Ok(())
    }
    pub fn refresh_sfen(&mut self, sfen: &str) -> Result<()> {
        if self.game != spec::GAME_SHOGI {
            return Err(RuntimeError::InvalidState("sfen refresh on non-shogi model".to_string()));
        }
        let b = crate::shogi::ShogiBoard::parse_sfen(sfen)?;
        let f = crate::shogi::extract_features(&b);
        self.acc.refresh(&self.tables, &f);
        self.feats = f;
        self.ctx = crate::shogi::compute_context(&b);
        self.phase = crate::shogi::game_phase(b.hand_total(), b.promo_count());
        Ok(())
    }
    pub fn refresh_xiangqi(&mut self, fen: &str) -> Result<()> {
        if self.game != spec::GAME_XIANGQI {
            return Err(RuntimeError::InvalidState("xiangqi refresh on non-xiangqi model".to_string()));
        }
        let b = crate::xiangqi::XiangqiBoard::parse_fen(fen)?;
        let f = crate::xiangqi::extract_features(&b);
        self.acc.refresh(&self.tables, &f);
        self.feats = f;
        self.ctx = crate::xiangqi::compute_context(&b);
        self.phase = crate::xiangqi::game_phase(&b);
        Ok(())
    }
    pub fn update_incremental(&mut self, before: &[(u8, u16)], after: &[(u8, u16)], ctx: &[f32]) {
        let (added, removed) = diff_features(before, after);
        if added.len() + removed.len() > FULL_REFRESH_LIMIT {
            self.acc.refresh(&self.tables, after);
            self.feats = after.to_vec();
            self.ctx = ctx.to_vec();
            return;
        }
        self.acc.apply_diff(&self.tables, &added, &removed);
        self.feats = after.to_vec();
        self.ctx = ctx.to_vec();
    }
    fn head_for(&self, phase: u8) -> &HeadWeights {
        let b = (phase as usize).min(self.heads.len().saturating_sub(1));
        &self.heads[b]
    }
    pub fn push(&mut self) {
        self.stack.push((self.acc.snapshot(), self.feats.clone(), self.ctx.clone(), self.phase));
    }
    pub fn pop(&mut self) -> bool {
        let entry = match self.stack.pop() {
            Some(v) => v,
            None => return false,
        };
        let (snap, feats, ctx, phase) = entry;
        self.acc.restore(&snap);
        self.feats = feats;
        self.ctx = ctx;
        self.phase = phase;
        true
    }
    pub fn search_depth(&self) -> usize {
        self.stack.len()
    }
    pub fn evaluate(&self) -> EvalResult {
        let mut tok = vec![0.0_f32; self.tokens * self.dim];
        self.acc.tokens(&mut tok);
        self.forward_tokens(&tok)
    }
    pub fn evaluate_board(&mut self, board: &Board) -> Result<EvalResult> {
        self.refresh(board)?;
        Ok(self.evaluate())
    }
    pub fn evaluate_sfen(&mut self, sfen: &str) -> Result<EvalResult> {
        self.refresh_sfen(sfen)?;
        Ok(self.evaluate())
    }
    pub fn evaluate_xiangqi(&mut self, fen: &str) -> Result<EvalResult> {
        self.refresh_xiangqi(fen)?;
        Ok(self.evaluate())
    }
    pub fn forward_tokens(&self, tok: &[f32]) -> EvalResult {
        let mut mixed = tok.to_vec();
        if let Some(mx) = &self.mixer {
            let mut out = vec![0.0_f32; self.tokens * self.dim];
            mx.forward(tok, self.active_ctx(), &mut out, None);
            mixed = out;
        }
        if let Some(du) = &self.dual {
            let mut out = vec![0.0_f32; self.tokens * self.dim];
            du.forward(&mixed, &mut out);
            mixed = out;
        }
        if let Some(mh) = &self.mh {
            let mut out = vec![0.0_f32; self.tokens * self.dim];
            mh.forward(tok, &mut out);
            mixed = out;
        }
        let (value, wdl, _) = self.head_for(self.phase).forward(&mixed);
        EvalResult { value, wdl, refine: false, difficulty: 0.0, policy: Vec::new(), score_mean: 0.0 }
    }
    pub fn evaluate_value_only(&self) -> f32 {
        let mut tok = vec![0.0_f32; self.tokens * self.dim];
        self.acc.tokens(&mut tok);
        let mut mixed = tok.to_vec();
        if let Some(mx) = &self.mixer {
            let mut out = vec![0.0_f32; self.tokens * self.dim];
            mx.forward(&tok, self.active_ctx(), &mut out, None);
            mixed = out;
        }
        if let Some(du) = &self.dual {
            let mut out = vec![0.0_f32; self.tokens * self.dim];
            du.forward(&mixed, &mut out);
            mixed = out;
        }
        if let Some(mh) = &self.mh {
            let mut out = vec![0.0_f32; self.tokens * self.dim];
            mh.forward(&tok, &mut out);
            mixed = out;
        }
        self.head_for(self.phase).forward_value_only(&mixed)
    }
    pub fn trace(&self) -> FullTrace {
        let mut tok = vec![0.0_f32; self.tokens * self.dim];
        self.acc.tokens(&mut tok);
        let mut tr = FullTrace {
            features: self.feats.clone(),
            accumulator: self.acc.raw().to_vec(),
            tokens: tok.clone(),
            ..Default::default()
        };
        let mut mixed = tok.clone();
        if let Some(mx) = &self.mixer {
            let mut out = vec![0.0_f32; self.tokens * self.dim];
            let mut mtr = MixerTrace::default();
            mx.forward(&tok, self.active_ctx(), &mut out, Some(&mut mtr));
            mixed = out;
            tr.mixer = mtr;
        }
        let (v, w, h) = self.head_for(self.phase).forward(&mixed);
        tr.head = h;
        tr.value = v;
        tr.wdl = w;
        tr
    }
    pub fn current_features(&self) -> &[(u8, u16)] {
        &self.feats
    }
    pub fn dump_text(&self) -> String {
        let tr = self.trace();
        let mut s = String::new();
        s.push_str(&format!("features {}\n", tr.features.len()));
        for (g, i) in &tr.features {
            s.push_str(&format!("f {} {}\n", g, i));
        }
        s.push_str(&format!("tokens {}\n", tr.tokens.len()));
        for v in &tr.tokens {
            s.push_str(&format!("t {:.9}\n", v));
        }
        s.push_str(&format!("value {:.9}\n", tr.value));
        s.push_str(&format!("wdl {:.9} {:.9} {:.9}\n", tr.wdl[0], tr.wdl[1], tr.wdl[2]));
        s
    }
    pub fn kernel_path(&self) -> &'static str {
        kernel::active_path_name()
    }
}
