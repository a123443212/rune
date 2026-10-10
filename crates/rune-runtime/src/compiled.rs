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

use rune_kernel::{arena::Arena, fused, Gate};
use crate::accumulator::{token_of, Tables};
use crate::board::Board;
use crate::compiled_loader::CompiledHeader;
use crate::error::{Result, RuntimeError};
use crate::evaluator::FULL_REFRESH_LIMIT;
use crate::features::{compute_context, extract_features, CONTEXT_DIM};
use crate::mixer::{dyn_factors, HeadWeights, MixerWeights};
use crate::mixer_dual::DualMixer;
use crate::mixer_mh::MultiHeadMixer;
use crate::mixer_soft::{scale_for_dim, soft_forward_into};
use crate::model_config::{resolve_model_config, ExecutionTarget};
use rune_model as model;
use std::path::Path;

pub struct CompiledEvaluator {
    tables: Tables,
    wq: Vec<f32>,
    bq: Vec<f32>,
    wk: Vec<f32>,
    bk: Vec<f32>,
    wv: Vec<f32>,
    bv: Vec<f32>,
    gab: Vec<f32>,
    dyn_u: Vec<f32>,
    dyn_w: Vec<f32>,
    ctx_dim: usize,
    gate: Gate,
    alpha: f32,
    soft: bool,
    soft_scale: f32,
    mh: Option<MultiHeadMixer>,
    dual: Option<DualMixer>,
    head: Vec<HeadWeights>,
    phase: u8,
    tokens: usize,
    dim: usize,
    acc: Vec<f32>,
    feats: Vec<(u8, u16)>,
    ctx: Vec<f32>,
    stack: Vec<(Vec<f32>, Vec<(u8, u16)>, Vec<f32>, u8)>,
    arena: Arena,
    header: CompiledHeader,
    arch_id: String,
}

fn need_vec(arrays: &std::collections::HashMap<String, Vec<f32>>, name: &str) -> Result<Vec<f32>> {
    arrays.get(name).cloned().ok_or_else(|| RuntimeError::TensorMissing(name.to_string()))
}

impl CompiledEvaluator {
    pub fn load(path: &Path) -> Result<CompiledEvaluator> {
        let header = crate::compiled_loader::load_compiled_header(path)?;
        let m = model::load(path).map_err(|e| RuntimeError::InvalidState(e.to_string()))?;
        if m.header.model_hash != header.model_hash && !header.model_hash.is_empty() {
            return Err(RuntimeError::InvalidState("model hash mismatch".to_string()));
        }
        Self::from_model(&m, header)
    }

    pub fn from_model(m: &model::RuneModel, header: CompiledHeader) -> Result<CompiledEvaluator> {
        let config = resolve_model_config(m, ExecutionTarget::Compiled)?;
        let arch = config.architecture_id;
        let tokens = config.tokens;
        let dim = config.dim;
        let vocabs = config.vocabs;
        let mut tables = Tables::zeros(dim, vocabs);
        for g in 0..9 {
            let k = format!("emb{}", g);
            let arr = m.arrays.get(&k).ok_or_else(|| RuntimeError::TensorMissing(k.clone()))?;
            if arr.len() != vocabs[g] * dim {
                return Err(RuntimeError::Shape(k));
            }
            tables.data[g].copy_from_slice(arr);
        }
        let has_mixer = arch == "RUNE-ATTN" || arch == "RUNE-ATTN-GAB" || arch == "RUNE-ATTN-SOFT" || arch == "RUNE-REL-02" || arch == "RUNE-ATTN-DUAL";
        let (wq, bq, wk, bk, wv, bv, gab, dyn_u, dyn_w, ctx_dim, gate, alpha);
        if has_mixer {
            wq = need_vec(&m.arrays, "wq")?;
            bq = need_vec(&m.arrays, "bq")?;
            wk = need_vec(&m.arrays, "wk")?;
            bk = need_vec(&m.arrays, "bk")?;
            wv = need_vec(&m.arrays, "wvv").or_else(|_| need_vec(&m.arrays, "wv"))?;
            bv = need_vec(&m.arrays, "bvv").or_else(|_| need_vec(&m.arrays, "bv"))?;
            if wq.len() != dim * dim || wk.len() != dim * dim || wv.len() != dim * dim {
                return Err(RuntimeError::Shape("mixer mat".to_string()));
            }
            if bq.len() != dim || bk.len() != dim || bv.len() != dim {
                return Err(RuntimeError::Shape("mixer bias".to_string()));
            }
            let gk = if m.arrays.contains_key("gabS") { "gabS" } else { "gab" };
            gab = match m.arrays.get(gk) {
                Some(v) => v.clone(),
                None if arch == "RUNE-ATTN" => vec![0.0; tokens * tokens],
                None => return Err(RuntimeError::TensorMissing(gk.to_string())),
            };
            if gab.len() != tokens * tokens {
                return Err(RuntimeError::Shape("gab".to_string()));
            }
            let pair = if arch == "RUNE-REL-02" {
                let cd = m.header.raw.get("context_dim").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                match (m.arrays.get("dynU"), m.arrays.get("dynW")) {
                    (Some(u), Some(w)) => {
                        if cd == 0 || cd != CONTEXT_DIM || u.len() != tokens * cd || w.len() != tokens * cd {
                            return Err(RuntimeError::Shape("dyn".to_string()));
                        }
                        (u.clone(), w.clone(), cd)
                    }
                    (None, None) => (Vec::new(), Vec::new(), 0),
                    _ => return Err(RuntimeError::Shape("dyn pair".to_string())),
                }
            } else {
                (Vec::new(), Vec::new(), 0)
            };
            dyn_u = pair.0;
            dyn_w = pair.1;
            ctx_dim = pair.2;
            gate = match m.header.raw.get("gate").and_then(|x| x.as_str()) {
                None | Some("softmax") => Gate::Clip,
                Some(s) => Gate::from_str(s).ok_or_else(|| RuntimeError::InvalidState("unknown gate".to_string()))?,
            };
            alpha = m.header.raw.get("alpha").and_then(|x| x.as_f64()).unwrap_or(1.0) as f32;
        } else {
            let n = dim * dim;
            wq = vec![0.0; n];
            bq = vec![0.0; dim];
            wk = vec![0.0; n];
            bk = vec![0.0; dim];
            wv = vec![0.0; n];
            bv = vec![0.0; dim];
            gab = vec![0.0; tokens * tokens];
            dyn_u = Vec::new();
            dyn_w = Vec::new();
            ctx_dim = 0;
            gate = Gate::Clip;
            alpha = 1.0;
        }
        let buckets = m.header.raw.get("head_buckets").and_then(|x| x.as_u64()).unwrap_or(1);
        if buckets != 1 && buckets != 3 {
            return Err(RuntimeError::InvalidState("unsupported head_buckets (want 1 or 3)".to_string()));
        }
        let mut heads: Vec<HeadWeights> = Vec::new();
        for b in 0..buckets {
            let suffix = if buckets == 1 { String::new() } else { format!("_b{}", b) };
            let is_swiglu = m.arrays.contains_key(&format!("wgate{}", suffix));
            let w1 = if is_swiglu { Vec::new() } else { need_vec(&m.arrays, &format!("w1{}", suffix))? };
            let b1 = if is_swiglu { Vec::new() } else { need_vec(&m.arrays, &format!("b1{}", suffix))? };
            let w2 = need_vec(&m.arrays, &format!("w2{}", suffix))?;
            let b2 = need_vec(&m.arrays, &format!("b2{}", suffix))?;
            let wvo = need_vec(&m.arrays, &format!("wvo{}", suffix))
                .or_else(|_| need_vec(&m.arrays, &format!("wv{}", suffix)))?;
            let bvo = m.arrays.get(&format!("bvo{}", suffix))
                .or_else(|| m.arrays.get(&format!("bv{}", suffix)))
                .and_then(|v| v.first().copied()).unwrap_or(0.0);
            let wwdl = need_vec(&m.arrays, &format!("wwdl{}", suffix))?;
            let bwdl = need_vec(&m.arrays, &format!("bwdl{}", suffix))?;
            let h2 = b2.len();
            let swiglu = if is_swiglu {
                let wgate = need_vec(&m.arrays, &format!("wgate{}", suffix))?;
                let bgate = need_vec(&m.arrays, &format!("bgate{}", suffix))?;
                let wup = need_vec(&m.arrays, &format!("wup{}", suffix))?;
                let bup = need_vec(&m.arrays, &format!("bup{}", suffix))?;
                let h1 = bgate.len();
                if bup.len() != h1 {
                    return Err(RuntimeError::Shape("swiglu bias".to_string()));
                }
                if wgate.len() != h1 * tokens * dim || wup.len() != h1 * tokens * dim {
                    return Err(RuntimeError::Shape("swiglu mat".to_string()));
                }
                if w2.len() != h2 * h1 {
                    return Err(RuntimeError::Shape("head mat".to_string()));
                }
                Some(crate::head_swiglu::SwiGluHead {
                    input: tokens * dim,
                    h1,
                    h2,
                    wgate,
                    bgate,
                    wup,
                    bup,
                    w2: w2.clone(),
                    b2: b2.clone(),
                    wvo: wvo.clone(),
                    bvo,
                    wwdl: wwdl.clone(),
                    bwdl: bwdl.clone(),
                })
            } else {
                None
            };
            let h1 = if is_swiglu { swiglu.as_ref().map(|s| s.h1).unwrap_or(0) } else { b1.len() };
            if !is_swiglu && (w1.len() != h1 * tokens * dim || w2.len() != h2 * h1) {
                return Err(RuntimeError::Shape("head mat".to_string()));
            }
            if wvo.len() != h2 {
                return Err(RuntimeError::Shape("wvo".to_string()));
            }
            if wwdl.len() != 3 * h2 || bwdl.len() != 3 {
                return Err(RuntimeError::Shape("wdl".to_string()));
            }
            heads.push(HeadWeights { input: tokens * dim, h1, h2, w1, b1, w2, b2, wvo, bvo, wwdl, bwdl, swiglu: swiglu });
        }
        let mh = if arch == "RUNE-ATTN-MH4" {
            if tokens != 8 || dim != 32 {
                return Err(RuntimeError::Shape("tokens".to_string()));
            }
            let mh_gate = match m.header.raw.get("gate").and_then(|x| x.as_str()) {
                None => Gate::Clip,
                Some(s) => Gate::from_str(s).ok_or_else(|| RuntimeError::InvalidState("unknown gate".to_string()))?,
            };
            let mut wq = Vec::new();
            let mut bq = Vec::new();
            let mut wk = Vec::new();
            let mut bk = Vec::new();
            let mut wv = Vec::new();
            let mut bv = Vec::new();
            let mut gab = Vec::new();
            for h in 0..4 {
                let s = format!("_h{}", h);
                wq.push(need_vec(&m.arrays, &format!("wq{}", s))?);
                bq.push(need_vec(&m.arrays, &format!("bq{}", s))?);
                wk.push(need_vec(&m.arrays, &format!("wk{}", s))?);
                bk.push(need_vec(&m.arrays, &format!("bk{}", s))?);
                wv.push(need_vec(&m.arrays, &format!("wv{}", s))?);
                bv.push(need_vec(&m.arrays, &format!("bv{}", s))?);
                gab.push(need_vec(&m.arrays, &format!("gab{}", s))?);
            }
            for h in 0..4 {
                if wq[h].len() != 8 * 32 || bq[h].len() != 8 || wk[h].len() != 8 * 32 {
                    return Err(RuntimeError::Shape("mh mat".to_string()));
                }
                if bk[h].len() != 8 || wv[h].len() != 8 * 32 || bv[h].len() != 8 {
                    return Err(RuntimeError::Shape("mh mat".to_string()));
                }
                if gab[h].len() != 64 {
                    return Err(RuntimeError::Shape("mh mat".to_string()));
                }
            }
            let wo = need_vec(&m.arrays, "wo")?;
            let bwo = need_vec(&m.arrays, "bwo")?;
            if wo.len() != 32 * 32 || bwo.len() != 32 {
                return Err(RuntimeError::Shape("mh wo".to_string()));
            }
            Some(MultiHeadMixer {
                heads: 4,
                head_dim: 8,
                tokens,
                dim,
                wq,
                bq,
                wk,
                bk,
                wv,
                bv,
                gab,
                wo,
                bwo,
                gate: mh_gate,
            })
        } else {
            None
        };
        let dual = if arch == "RUNE-ATTN-DUAL" {
            let wq2 = need_vec(&m.arrays, "wq2")?;
            let bq2 = need_vec(&m.arrays, "bq2")?;
            let wk2 = need_vec(&m.arrays, "wk2")?;
            let bk2 = need_vec(&m.arrays, "bk2")?;
            let wv2 = need_vec(&m.arrays, "wvv2").or_else(|_| need_vec(&m.arrays, "wv2"))?;
            let bv2 = need_vec(&m.arrays, "bvv2").or_else(|_| need_vec(&m.arrays, "bv2"))?;
            let gab2 = match m.arrays.get("gab2") {
                Some(v) => v.clone(),
                None => vec![0.0; tokens * tokens],
            };
            let first = MixerWeights {
                tokens,
                dim,
                wq: wq.clone(),
                bq: bq.clone(),
                wk: wk.clone(),
                bk: bk.clone(),
                wv: wv.clone(),
                bv: bv.clone(),
                gab: gab.clone(),
                dyn_u: Vec::new(),
                dyn_w: Vec::new(),
                ctx_dim: 0,
                gate,
                alpha,
            };
            let second = MixerWeights {
                tokens,
                dim,
                wq: wq2,
                bq: bq2,
                wk: wk2,
                bk: bk2,
                wv: wv2,
                bv: bv2,
                gab: gab2,
                dyn_u: Vec::new(),
                dyn_w: Vec::new(),
                ctx_dim: 0,
                gate,
                alpha: 1.0,
            };
            Some(DualMixer { first, second })
        } else {
            None
        };
        let arena = Arena::new(header.memory_bytes.max(8192));
        Ok(CompiledEvaluator {
            tables,
            wq,
            bq,
            wk,
            bk,
            wv,
            bv,
            gab,
            dyn_u,
            dyn_w,
            ctx_dim,
            gate,
            alpha,
            soft: arch == "RUNE-ATTN-SOFT",
            soft_scale: scale_for_dim(dim),
            mh,
            dual,
            head: heads,
            phase: 1,
            tokens,
            dim,
            acc: vec![0.0; tokens * dim],
            feats: Vec::new(),
            ctx: Vec::new(),
            stack: Vec::new(),
            arena,
            header,
            arch_id: arch,
        })
    }

    pub fn arch_id(&self) -> &str {
        &self.arch_id
    }

    pub fn target_isa(&self) -> &str {
        &self.header.target_isa
    }

    pub fn arena_bytes(&self) -> usize {
        self.arena.bytes()
    }

    fn head_for(&self, phase: u8) -> &HeadWeights {
        let b = (phase as usize).min(self.head.len().saturating_sub(1));
        &self.head[b]
    }
    pub fn refresh(&mut self, board: &Board) {
        let f = extract_features(board);
        self.refresh_features(&f);
        self.ctx = compute_context(board);
        self.phase = board.game_phase();
    }

    pub fn refresh_features(&mut self, feats: &[(u8, u16)]) {
        for v in self.acc.iter_mut() {
            *v = 0.0;
        }
        self.apply_add(feats);
        self.feats = feats.to_vec();
    }

    fn apply_add(&mut self, feats: &[(u8, u16)]) {
        let dim = self.dim;
        let tokens = self.tokens;
        for (g, idx) in feats {
            let t = match token_of(tokens, *g, *idx) {
                Some(v) => v,
                None => continue,
            };
            let base = t * dim;
            let row = self.tables.row(*g as usize, *idx as usize);
            for d in 0..dim {
                self.acc[base + d] += row[d];
            }
        }
    }

    fn apply_remove(&mut self, feats: &[(u8, u16)]) {
        let dim = self.dim;
        let tokens = self.tokens;
        for (g, idx) in feats {
            let t = match token_of(tokens, *g, *idx) {
                Some(v) => v,
                None => continue,
            };
            let base = t * dim;
            let row = self.tables.row(*g as usize, *idx as usize);
            for d in 0..dim {
                self.acc[base + d] -= row[d];
            }
        }
    }

    fn dyn_on(&self) -> bool {
        !self.dyn_u.is_empty() && !self.dyn_w.is_empty() && self.ctx_dim > 0 && !self.ctx.is_empty()
    }

    fn mix_into(&self, tok: &[f32], mixed: &mut [f32]) {
        let t = self.tokens;
        let d = self.dim;
        let mut q = vec![0.0f32; t * d];
        let mut k = vec![0.0f32; t * d];
        let mut vv = vec![0.0f32; t * d];
        for i in 0..t {
            let xb = &tok[i * d..(i + 1) * d];
            let qb = &mut q[i * d..(i + 1) * d];
            let kb = &mut k[i * d..(i + 1) * d];
            let vb = &mut vv[i * d..(i + 1) * d];
            rune_kernel::mat_vec(&self.wq, xb, Some(&self.bq), qb, d, d);
            rune_kernel::mat_vec(&self.wk, xb, Some(&self.bk), kb, d, d);
            rune_kernel::mat_vec(&self.wv, xb, Some(&self.bv), vb, d, d);
        }
        let mut s = vec![0.0f32; t * t];
        rune_kernel::mat_mul_tt(&q, &k, &mut s, t, t, d);
        let ctx = if self.ctx.is_empty() { None } else { Some(self.ctx.as_slice()) };
        let (du, dw) = dyn_factors(&self.dyn_u, &self.dyn_w, t, self.ctx_dim, ctx);
        let dyn_on = self.dyn_on();
        let mut g = vec![0.0f32; t * t];
        for a in 0..t {
            for b in 0..t {
                let mut v = s[a * t + b] + self.gab[a * t + b];
                if dyn_on {
                    v += rune_kernel::clamp_delta(du[a] * dw[b]);
                }
                g[a * t + b] = self.gate.apply(v);
            }
        }
        let mut y = vec![0.0f32; t * d];
        rune_kernel::mat_mul(&g, &vv, &mut y, t, d, t);
        for i in 0..t * d {
            mixed[i] = tok[i] + self.alpha * y[i];
        }
    }

    pub fn update_incremental(&mut self, before: &[(u8, u16)], after: &[(u8, u16)], ctx: &[f32]) {
        let (added, removed) = crate::features::diff_features(before, after);
        if added.len() + removed.len() > FULL_REFRESH_LIMIT {
            self.refresh_features(after);
            self.feats = after.to_vec();
            self.ctx = ctx.to_vec();
            return;
        }
        self.apply_add(&added);
        self.apply_remove(&removed);
        self.feats = after.to_vec();
        self.ctx = ctx.to_vec();
    }

    fn tokens_into(&self, out: &mut [f32]) {
        for i in 0..out.len() {
            let v = self.acc[i];
            out[i] = if v < 0.0 { 0.0 } else if v > 1.0 { 1.0 } else { v };
        }
    }

    pub fn evaluate(&mut self) -> crate::evaluator::EvalResult {
        let tokens = self.tokens;
        let dim = self.dim;
        let flat_n = tokens * dim;
        let mut tok = vec![0.0f32; flat_n];
        self.tokens_into(&mut tok);
        self.forward_tokens_compiled(&tok)
    }

    pub fn evaluate_board(&mut self, board: &Board) -> crate::evaluator::EvalResult {
        self.refresh(board);
        self.evaluate()
    }
    pub fn push(&mut self) {
        self.stack.push((self.acc.clone(), self.feats.clone(), self.ctx.clone(), self.phase));
    }
    pub fn pop(&mut self) -> bool {
        let entry = match self.stack.pop() {
            Some(v) => v,
            None => return false,
        };
        let (snap, feats, ctx, phase) = entry;
        self.acc = snap;
        self.feats = feats;
        self.ctx = ctx;
        self.phase = phase;
        true
    }
    pub fn search_depth(&self) -> usize {
        self.stack.len()
    }
    pub fn evaluate_value_only(&self) -> f32 {
        let flat_n = self.tokens * self.dim;
        let mut tok = vec![0.0f32; flat_n];
        self.tokens_into(&mut tok);
        let t = self.tokens;
        let d = self.dim;
        let is_soft = self.soft;
        let has_mixer = self.arch_id == "RUNE-ATTN" || self.arch_id == "RUNE-ATTN-GAB" || self.arch_id == "RUNE-ATTN-SOFT" || self.arch_id == "RUNE-REL-02" || self.arch_id == "RUNE-ATTN-DUAL";
        let mut mixed = tok.to_vec();
        if is_soft {
            let mut out = vec![0.0f32; self.tokens * self.dim];
            soft_forward_into(&self.wq, &self.bq, &self.wk, &self.bk, &self.wv, &self.bv, &self.gab, self.soft_scale, t, d, &tok, &mut out);
            mixed = out;
        } else if has_mixer && t == 8 && d == 32 && !self.dyn_on() && self.dual.is_none() {
            let mut q = vec![0.0f32; 256];
            let mut k = vec![0.0f32; 256];
            let mut vv = vec![0.0f32; 256];
            let mut s = vec![0.0f32; 64];
            let mut g = vec![0.0f32; 64];
            let mut tmp = vec![0.0f32; 256];
            let mut out = vec![0.0f32; 256];
            fused::qkv_fused_8x32(&self.wq, &self.bq, &self.wk, &self.bk, &self.wv, &self.bv, &tok, &mut q, &mut k, &mut vv);
            fused::score_bias_gate_8x8(&q, &k, &self.gab, self.gate, &mut s, &mut g);
            fused::mix_residual_8x32(&g, &vv, &tok, self.alpha, &mut tmp, &mut out);
            mixed = out;
        } else if has_mixer {
            self.mix_into(&tok, &mut mixed);
        }
        if let Some(du) = &self.dual {
            let mut out = vec![0.0f32; self.tokens * self.dim];
            du.forward(&mixed, &mut out);
            mixed = out;
        }
        if let Some(mh) = &self.mh {
            let mut out = vec![0.0f32; self.tokens * self.dim];
            mh.forward(&tok, &mut out);
            mixed = out;
        }
        self.head_for(self.phase).forward_value_only(&mixed)
    }

    pub fn forward_tokens_compiled(&mut self, tok: &[f32]) -> crate::evaluator::EvalResult {
        let t = self.tokens;
        let d = self.dim;
        let is_soft = self.soft;
        let has_mixer = self.arch_id == "RUNE-ATTN" || self.arch_id == "RUNE-ATTN-GAB" || self.arch_id == "RUNE-ATTN-SOFT" || self.arch_id == "RUNE-REL-02" || self.arch_id == "RUNE-ATTN-DUAL";
        let mut mixed = tok.to_vec();
        if is_soft {
            let mut out = vec![0.0f32; self.tokens * self.dim];
            soft_forward_into(&self.wq, &self.bq, &self.wk, &self.bk, &self.wv, &self.bv, &self.gab, self.soft_scale, t, d, tok, &mut out);
            mixed = out;
        } else if has_mixer && t == 8 && d == 32 && !self.dyn_on() && self.dual.is_none() {
            let mut q = vec![0.0f32; 256];
            let mut k = vec![0.0f32; 256];
            let mut vv = vec![0.0f32; 256];
            let mut s = vec![0.0f32; 64];
            let mut g = vec![0.0f32; 64];
            let mut tmp = vec![0.0f32; 256];
            let mut out = vec![0.0f32; 256];
            fused::qkv_fused_8x32(&self.wq, &self.bq, &self.wk, &self.bk, &self.wv, &self.bv, tok, &mut q, &mut k, &mut vv);
            fused::score_bias_gate_8x8(&q, &k, &self.gab, self.gate, &mut s, &mut g);
            fused::mix_residual_8x32(&g, &vv, tok, self.alpha, &mut tmp, &mut out);
            mixed = out;
        } else if has_mixer {
            self.mix_into(tok, &mut mixed);
        }
        if let Some(du) = &self.dual {
            let mut out = vec![0.0f32; self.tokens * self.dim];
            du.forward(&mixed, &mut out);
            mixed = out;
        }
        if let Some(mh) = &self.mh {
            let mut out = vec![0.0f32; self.tokens * self.dim];
            mh.forward(tok, &mut out);
            mixed = out;
        }
        let (value, wdl, _) = self.head_for(self.phase).forward(&mixed);
        crate::evaluator::EvalResult { value, wdl, refine: false, difficulty: 0.0, policy: Vec::new(), score_mean: 0.0 }
    }
}
