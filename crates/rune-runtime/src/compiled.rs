use rune_kernel::{arena::Arena, fused};
use crate::accumulator::{token_of, Tables};
use crate::board::Board;
use crate::compiled_loader::CompiledHeader;
use crate::error::{Result, RuntimeError};
use crate::evaluator::FULL_REFRESH_LIMIT;
use crate::features::{compute_context, extract_features, CONTEXT_DIM};
use crate::mixer::{dyn_factors, HeadWeights};
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
    gate_hard: bool,
    alpha: f32,
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
        let arch = m.header.architecture_id.clone();
        if arch != "RUNE-SFNN" && arch != "RUNE-MLP" && arch != "RUNE-ATTN" && arch != "RUNE-ATTN-GAB" && arch != "RUNE-REL-02" {
            return Err(RuntimeError::UnsupportedArch(arch));
        }
        let tokens = m.header.tokens;
        let dim = m.header.token_dim;
        let flex = arch == "RUNE-REL-02";
        if flex {
            if tokens != 6 && tokens != 8 && tokens != 10 {
                return Err(RuntimeError::Shape("tokens".to_string()));
            }
            if dim != 24 && dim != 32 && dim != 40 {
                return Err(RuntimeError::Shape("token dim".to_string()));
            }
        } else if tokens != 8 || dim != 32 {
            return Err(RuntimeError::Shape("tokens".to_string()));
        }
        let vocabs = rune_spec::VOCAB_SIZES;
        let mut tables = Tables::zeros(dim, vocabs);
        for g in 0..9 {
            let k = format!("emb{}", g);
            let arr = m.arrays.get(&k).ok_or_else(|| RuntimeError::TensorMissing(k.clone()))?;
            if arr.len() != vocabs[g] * dim {
                return Err(RuntimeError::Shape(k));
            }
            tables.data[g].copy_from_slice(arr);
        }
        let has_mixer = arch == "RUNE-ATTN" || arch == "RUNE-ATTN-GAB" || arch == "RUNE-REL-02";
        let (wq, bq, wk, bk, wv, bv, gab, dyn_u, dyn_w, ctx_dim, gate_hard, alpha);
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
            gate_hard = m.header.raw.get("gate").and_then(|x| x.as_str()).unwrap_or("clip") == "hard_sigmoid";
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
            gate_hard = false;
            alpha = 1.0;
        }
        let buckets = m.header.raw.get("head_buckets").and_then(|x| x.as_u64()).unwrap_or(1);
        if buckets != 1 && buckets != 3 {
            return Err(RuntimeError::InvalidState("unsupported head_buckets (want 1 or 3)".to_string()));
        }
        let mut heads: Vec<HeadWeights> = Vec::new();
        for b in 0..buckets {
            let suffix = if buckets == 1 { String::new() } else { format!("_b{}", b) };
            let w1 = need_vec(&m.arrays, &format!("w1{}", suffix))?;
            let b1 = need_vec(&m.arrays, &format!("b1{}", suffix))?;
            let w2 = need_vec(&m.arrays, &format!("w2{}", suffix))?;
            let b2 = need_vec(&m.arrays, &format!("b2{}", suffix))?;
            let wvo = need_vec(&m.arrays, &format!("wvo{}", suffix))
                .or_else(|_| need_vec(&m.arrays, &format!("wv{}", suffix)))?;
            let bvo = m.arrays.get(&format!("bvo{}", suffix))
                .or_else(|| m.arrays.get(&format!("bv{}", suffix)))
                .and_then(|v| v.first().copied()).unwrap_or(0.0);
            let wwdl = need_vec(&m.arrays, &format!("wwdl{}", suffix))?;
            let bwdl = need_vec(&m.arrays, &format!("bwdl{}", suffix))?;
            let h1 = b1.len();
            let h2 = b2.len();
            if w1.len() != h1 * tokens * dim || w2.len() != h2 * h1 {
                return Err(RuntimeError::Shape("head mat".to_string()));
            }
            if wvo.len() != h2 {
                return Err(RuntimeError::Shape("wvo".to_string()));
            }
            if wwdl.len() != 3 * h2 || bwdl.len() != 3 {
                return Err(RuntimeError::Shape("wdl".to_string()));
            }
            heads.push(HeadWeights { input: tokens * dim, h1, h2, w1, b1, w2, b2, wvo, bvo, wwdl, bwdl });
        }
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
            gate_hard,
            alpha,
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
                g[a * t + b] = if self.gate_hard { rune_kernel::hard_sigmoid(v) } else { rune_kernel::clipped_relu(v) };
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
    pub fn pop(&mut self) {
        let (snap, feats, ctx, phase) = self.stack.pop().expect("pop without push");
        self.acc = snap;
        self.feats = feats;
        self.ctx = ctx;
        self.phase = phase;
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
        let has_mixer = self.arch_id == "RUNE-ATTN" || self.arch_id == "RUNE-ATTN-GAB" || self.arch_id == "RUNE-REL-02";
        let mut mixed = tok.to_vec();
        if has_mixer && t == 8 && d == 32 && !self.dyn_on() {
            let mut q = vec![0.0f32; 256];
            let mut k = vec![0.0f32; 256];
            let mut vv = vec![0.0f32; 256];
            let mut s = vec![0.0f32; 64];
            let mut g = vec![0.0f32; 64];
            let mut tmp = vec![0.0f32; 256];
            let mut out = vec![0.0f32; 256];
            fused::qkv_fused_8x32(&self.wq, &self.bq, &self.wk, &self.bk, &self.wv, &self.bv, &tok, &mut q, &mut k, &mut vv);
            fused::score_bias_gate_8x8(&q, &k, &self.gab, self.gate_hard, &mut s, &mut g);
            fused::mix_residual_8x32(&g, &vv, &tok, self.alpha, &mut tmp, &mut out);
            mixed = out;
        } else if has_mixer {
            self.mix_into(&tok, &mut mixed);
        }
        self.head_for(self.phase).forward_value_only(&mixed)
    }

    pub fn forward_tokens_compiled(&mut self, tok: &[f32]) -> crate::evaluator::EvalResult {
        let t = self.tokens;
        let d = self.dim;
        let has_mixer = self.arch_id == "RUNE-ATTN" || self.arch_id == "RUNE-ATTN-GAB" || self.arch_id == "RUNE-REL-02";
        let mut mixed = tok.to_vec();
        if has_mixer && t == 8 && d == 32 && !self.dyn_on() {
            let mut q = vec![0.0f32; 256];
            let mut k = vec![0.0f32; 256];
            let mut vv = vec![0.0f32; 256];
            let mut s = vec![0.0f32; 64];
            let mut g = vec![0.0f32; 64];
            let mut tmp = vec![0.0f32; 256];
            let mut out = vec![0.0f32; 256];
            fused::qkv_fused_8x32(&self.wq, &self.bq, &self.wk, &self.bk, &self.wv, &self.bv, tok, &mut q, &mut k, &mut vv);
            fused::score_bias_gate_8x8(&q, &k, &self.gab, self.gate_hard, &mut s, &mut g);
            fused::mix_residual_8x32(&g, &vv, tok, self.alpha, &mut tmp, &mut out);
            mixed = out;
        } else if has_mixer {
            self.mix_into(&tok, &mut mixed);
        }
        let (value, wdl, _) = self.head_for(self.phase).forward(&mixed);
        crate::evaluator::EvalResult { value, wdl, refine: false, difficulty: 0.0 }
    }
}
