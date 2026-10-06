use std::collections::HashMap;
use std::path::Path;
use rune_kernel as kernel;
use rune_model as model;
use rune_spec as spec;
use crate::accumulator::{Accumulator, Tables};
use crate::board::Board;
use crate::error::{Result, RuntimeError};
use crate::features::{compute_context, diff_features, extract_features, CONTEXT_DIM};
use crate::mixer::{HeadTrace, HeadWeights, MixerTrace, MixerWeights};

pub(crate) const FULL_REFRESH_LIMIT: usize = 64;
#[derive(Debug, Clone)]
pub struct EvalResult {
    pub value: f32,
    pub wdl: [f32; 3],
    pub refine: bool,
    pub difficulty: f32,
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
    head: HeadWeights,
    arch_id: String,
    tokens: usize,
    dim: usize,
    acc: Accumulator,
    feats: Vec<(u8, u16)>,
    ctx: Vec<f32>,
    stack: Vec<(Vec<f32>, Vec<(u8, u16)>, Vec<f32>)>,
}
fn need_vec(arrays: &HashMap<String, Vec<f32>>, name: &str) -> Result<Vec<f32>> {
    arrays.get(name).cloned().ok_or_else(|| RuntimeError::TensorMissing(name.to_string()))
}
fn scalar_of(arrays: &HashMap<String, Vec<f32>>, name: &str) -> Result<f32> {
    let v = need_vec(arrays, name)?;
    if v.is_empty() {
        return Err(RuntimeError::Shape(name.to_string()));
    }
    Ok(v[0])
}
impl Evaluator {
    pub fn load(path: &Path) -> Result<Evaluator> {
        let m = model::load(path).map_err(|e| RuntimeError::InvalidState(e.to_string()))?;
        Evaluator::from_model(&m)
    }
    pub fn from_model(m: &model::RuneModel) -> Result<Evaluator> {
        if m.header.feature_version != spec::FEATURE_VERSION {
            return Err(RuntimeError::FeatureMismatch(m.header.feature_version.clone()));
        }
        let arch = m.header.architecture_id.clone();
        let supported = arch == "RUNE-SFNN"
            || arch == "RUNE-MLP"
            || arch == "RUNE-ATTN"
            || arch == "RUNE-ATTN-GAB"
            || arch == "RUNE-REL-02";
        if !supported {
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
        let vocabs = spec::VOCAB_SIZES;
        let mut tables = Tables::zeros(dim, vocabs);
        for g in 0..8 {
            let k = format!("emb{}", g);
            let arr = m.arrays.get(&k).ok_or_else(|| RuntimeError::TensorMissing(k.clone()))?;
            if arr.len() != vocabs[g] * dim {
                return Err(RuntimeError::Shape(k));
            }
            tables.data[g].copy_from_slice(arr);
        }
        let has_mixer = arch == "RUNE-ATTN" || arch == "RUNE-ATTN-GAB" || arch == "RUNE-REL-02";
        let mixer = if has_mixer {
            let wq = need_vec(&m.arrays, "wq")?;
            let bq = need_vec(&m.arrays, "bq")?;
            let wk = need_vec(&m.arrays, "wk")?;
            let bk = need_vec(&m.arrays, "bk")?;
            let wv = need_vec(&m.arrays, "wvv").or_else(|_| need_vec(&m.arrays, "wv"))?;
            let bv = need_vec(&m.arrays, "bvv").or_else(|_| need_vec(&m.arrays, "bv"))?;
            if wq.len() != dim * dim || wk.len() != dim * dim || wv.len() != dim * dim {
                return Err(RuntimeError::Shape("mixer mat".to_string()));
            }
            if bq.len() != dim || bk.len() != dim || bv.len() != dim {
                return Err(RuntimeError::Shape("mixer bias".to_string()));
            }
            let gab_key = if m.arrays.contains_key("gabS") { "gabS" } else { "gab" };
            let gab = match m.arrays.get(gab_key) {
                Some(v) => v.clone(),
                None if arch == "RUNE-ATTN" => vec![0.0; tokens * tokens],
                None => return Err(RuntimeError::TensorMissing(gab_key.to_string())),
            };
            if gab.len() != tokens * tokens {
                return Err(RuntimeError::Shape("gab".to_string()));
            }
            let gate_hard = m.header.raw.get("gate").and_then(|x| x.as_str()).unwrap_or("clip") == "hard_sigmoid";
            let alpha = m.header.raw.get("alpha").and_then(|x| x.as_f64()).unwrap_or(1.0) as f32;
            let (dyn_u, dyn_w, ctx_dim) = if arch == "RUNE-REL-02" {
                let cd = m.header.raw.get("context_dim").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                let pair = (m.arrays.get("dynU"), m.arrays.get("dynW"));
                match pair {
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
            Some(MixerWeights { tokens, dim, wq, bq, wk, bk, wv, bv, gab, dyn_u, dyn_w, ctx_dim, gate_hard, alpha })
        } else {
            None
        };
        let (h1n, h2n, w1n, b1n) = if arch == "RUNE-SFNN" {
            ("w1", "b1", 256usize, 256usize)
        } else {
            ("w1", "b1", 128usize, 128usize)
        };
        let _ = (h1n, b1n);
        let w1 = need_vec(&m.arrays, "w1")?;
        let b1 = need_vec(&m.arrays, "b1")?;
        let w2 = need_vec(&m.arrays, "w2")?;
        let b2 = need_vec(&m.arrays, "b2")?;
        let wvo = need_vec(&m.arrays, "wvo").or_else(|_| need_vec(&m.arrays, "wv"))?;
        let bvo = scalar_of(&m.arrays, "bvo").or_else(|_| scalar_of(&m.arrays, "bv"))?;
        let wwdl = need_vec(&m.arrays, "wwdl")?;
        let bwdl = need_vec(&m.arrays, "bwdl")?;
        let h1 = b1.len();
        let h2 = b2.len();
        let input = tokens * dim;
        if w1.len() != h1 * input || w2.len() != h2 * h1 {
            return Err(RuntimeError::Shape("head mat".to_string()));
        }
        if wvo.len() != h2 {
            return Err(RuntimeError::Shape("wvo".to_string()));
        }
        if wwdl.len() != 3 * h2 || bwdl.len() != 3 {
            return Err(RuntimeError::Shape("wdl".to_string()));
        }
        let _ = w1n;
        let _ = h2n;
        let head = HeadWeights { input, h1, h2, w1, b1, w2, b2, wvo, bvo, wwdl, bwdl };
        let acc = Accumulator::new(tokens, dim);
        Ok(Evaluator { tables, mixer, head, arch_id: arch, tokens, dim, acc, feats: Vec::new(), ctx: Vec::new(), stack: Vec::new() })
    }
    pub fn arch_id(&self) -> &str {
        &self.arch_id
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
    pub fn refresh(&mut self, board: &Board) {
        let f = extract_features(board);
        self.acc.refresh(&self.tables, &f);
        self.feats = f;
        self.ctx = compute_context(board);
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
    pub fn push(&mut self) {
        self.stack.push((self.acc.snapshot(), self.feats.clone(), self.ctx.clone()));
    }
    pub fn pop(&mut self) {
        let (snap, feats, ctx) = self.stack.pop().expect("pop without push");
        self.acc.restore(&snap);
        self.feats = feats;
        self.ctx = ctx;
    }
    pub fn search_depth(&self) -> usize {
        self.stack.len()
    }
    pub fn evaluate(&self) -> EvalResult {
        let mut tok = vec![0.0_f32; self.tokens * self.dim];
        self.acc.tokens(&mut tok);
        self.forward_tokens(&tok)
    }
    pub fn evaluate_board(&mut self, board: &Board) -> EvalResult {
        self.refresh(board);
        self.evaluate()
    }
    pub fn forward_tokens(&self, tok: &[f32]) -> EvalResult {
        let mut mixed = tok.to_vec();
        if let Some(mx) = &self.mixer {
            let mut out = vec![0.0_f32; self.tokens * self.dim];
            mx.forward(tok, self.active_ctx(), &mut out, None);
            mixed = out;
        }
        let (value, wdl, _) = self.head.forward(&mixed);
        EvalResult { value, wdl, refine: false, difficulty: 0.0 }
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
        self.head.forward_value_only(&mixed)
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
        let (v, w, h) = self.head.forward(&mixed);
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
