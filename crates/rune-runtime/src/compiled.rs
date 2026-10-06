use rune_kernel::{arena::Arena, fused};
use crate::accumulator::Tables;
use crate::board::Board;
use crate::compiled_loader::CompiledHeader;
use crate::error::{Result, RuntimeError};
use crate::features::extract_features;
use crate::mixer::HeadWeights;
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
    gate_hard: bool,
    alpha: f32,
    head: HeadWeights,
    tokens: usize,
    dim: usize,
    acc: Vec<f32>,
    feats: Vec<(u8, u16)>,
    stack: Vec<(Vec<f32>, Vec<(u8, u16)>)>,
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
        let vocabs = rune_spec::VOCAB_SIZES;
        let mut tables = Tables::zeros(dim, vocabs);
        for g in 0..8 {
            let k = format!("emb{}", g);
            let arr = m.arrays.get(&k).ok_or_else(|| RuntimeError::TensorMissing(k.clone()))?;
            tables.data[g].copy_from_slice(arr);
        }
        let has_mixer = arch == "RUNE-ATTN" || arch == "RUNE-ATTN-GAB" || arch == "RUNE-REL-02";
        let (wq, bq, wk, bk, wv, bv, gab, gate_hard, alpha);
        if has_mixer {
            wq = need_vec(&m.arrays, "wq")?;
            bq = need_vec(&m.arrays, "bq")?;
            wk = need_vec(&m.arrays, "wk")?;
            bk = need_vec(&m.arrays, "bk")?;
            wv = need_vec(&m.arrays, "wvv").or_else(|_| need_vec(&m.arrays, "wv"))?;
            bv = need_vec(&m.arrays, "bvv").or_else(|_| need_vec(&m.arrays, "bv"))?;
            let gk = if m.arrays.contains_key("gabS") { "gabS" } else { "gab" };
            gab = need_vec(&m.arrays, gk)?;
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
            gate_hard = false;
            alpha = 1.0;
        }
        let w1 = need_vec(&m.arrays, "w1")?;
        let b1 = need_vec(&m.arrays, "b1")?;
        let w2 = need_vec(&m.arrays, "w2")?;
        let b2 = need_vec(&m.arrays, "b2")?;
        let wvo = need_vec(&m.arrays, "wvo").or_else(|_| need_vec(&m.arrays, "wv"))?;
        let bvo = m.arrays.get("bvo").or_else(|| m.arrays.get("bv")).and_then(|v| v.first().copied()).unwrap_or(0.0);
        let wwdl = need_vec(&m.arrays, "wwdl")?;
        let bwdl = need_vec(&m.arrays, "bwdl")?;
        let h1 = b1.len();
        let h2 = b2.len();
        let head = HeadWeights { input: tokens * dim, h1, h2, w1, b1, w2, b2, wvo, bvo, wwdl, bwdl };
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
            gate_hard,
            alpha,
            head,
            tokens,
            dim,
            acc: vec![0.0; tokens * dim],
            feats: Vec::new(),
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

    pub fn refresh(&mut self, board: &Board) {
        let f = extract_features(board);
        self.refresh_features(&f);
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
        let mut by_group: [Vec<u16>; 8] = Default::default();
        for (g, idx) in feats {
            by_group[*g as usize].push(*idx);
        }
        for g in 0..8 {
            if by_group[g].is_empty() {
                continue;
            }
            let base = g * dim;
            for idx in &by_group[g] {
                let row = self.tables.row(g, *idx as usize);
                for d in 0..dim {
                    self.acc[base + d] += row[d];
                }
            }
        }
    }

    pub fn update_incremental(&mut self, before: &[(u8, u16)], after: &[(u8, u16)]) {
        let (added, removed) = crate::features::diff_features(before, after);
        if added.len() + removed.len() > crate::evaluator::FULL_REFRESH_LIMIT {
            self.refresh_features(after);
            return;
        }
        let dim = self.dim;
        for (g, idx) in &added {
            let row = self.tables.row(*g as usize, *idx as usize).to_vec();
            let base = *g as usize * dim;
            for d in 0..dim {
                self.acc[base + d] += row[d];
            }
        }
        for (g, idx) in &removed {
            let row = self.tables.row(*g as usize, *idx as usize).to_vec();
            let base = *g as usize * dim;
            for d in 0..dim {
                self.acc[base + d] -= row[d];
            }
        }
        self.feats = after.to_vec();
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
        self.stack.push((self.acc.clone(), self.feats.clone()));
    }
    pub fn pop(&mut self) {
        let (snap, feats) = self.stack.pop().expect("pop without push");
        self.acc = snap;
        self.feats = feats;
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
        if has_mixer {
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
            let mut g = vec![0.0f32; t * t];
            for a in 0..t {
                for b in 0..t {
                    let v = s[a * t + b] + self.gab[a * t + b];
                    g[a * t + b] = if self.gate_hard { rune_kernel::hard_sigmoid(v) } else { rune_kernel::clipped_relu(v) };
                }
            }
            let mut y = vec![0.0f32; t * d];
            rune_kernel::mat_mul(&g, &vv, &mut y, t, d, t);
            for i in 0..flat_n {
                mixed[i] = tok[i] + self.alpha * y[i];
            }
        }
        self.head.forward_value_only(&mixed)
    }

    pub fn forward_tokens_compiled(&mut self, tok: &[f32]) -> crate::evaluator::EvalResult {
        let t = self.tokens;
        let d = self.dim;
        let has_mixer = self.arch_id == "RUNE-ATTN" || self.arch_id == "RUNE-ATTN-GAB" || self.arch_id == "RUNE-REL-02";
        let mut mixed = tok.to_vec();
        if has_mixer && t == 8 && d == 32 {
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
            let mut g = vec![0.0f32; t * t];
            for a in 0..t {
                for b in 0..t {
                    let v = s[a * t + b] + self.gab[a * t + b];
                    g[a * t + b] = if self.gate_hard { rune_kernel::hard_sigmoid(v) } else { rune_kernel::clipped_relu(v) };
                }
            }
            let mut y = vec![0.0f32; t * d];
            rune_kernel::mat_mul(&g, &vv, &mut y, t, d, t);
            for i in 0..t * d {
                mixed[i] = tok[i] + self.alpha * y[i];
            }
        }
        let (value, wdl, _) = self.head.forward(&mixed);
        crate::evaluator::EvalResult { value, wdl, refine: false, difficulty: 0.0 }
    }
}
