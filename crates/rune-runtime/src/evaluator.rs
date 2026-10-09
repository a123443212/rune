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
use crate::mixer_mh::MultiHeadMixer;

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
    mh: Option<MultiHeadMixer>,
    heads: Vec<HeadWeights>,
    pub(crate) resnet: Option<crate::resnet::ResnetWeights>,
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
        let game = m.header.game.clone();
        let want_feat = spec::game_feature_version(&game).ok_or_else(|| RuntimeError::GameMismatch(game.clone()))?;
        if m.header.feature_version != want_feat {
            return Err(RuntimeError::FeatureMismatch(m.header.feature_version.clone()));
        }
        let arch = m.header.architecture_id.clone();
        let is_resnet = arch.starts_with("RUNE-RESNET");
        let supported = arch == "RUNE-SFNN"
            || arch == "RUNE-MLP"
            || arch == "RUNE-ATTN"
            || arch == "RUNE-ATTN-GAB"
            || arch == "RUNE-ATTN-MH4"
            || arch == "RUNE-REL-02"
            || is_resnet;
        if !supported {
            return Err(RuntimeError::UnsupportedArch(arch));
        }
        let tokens = m.header.tokens;
        let dim = m.header.token_dim;
        let flex = arch == "RUNE-REL-02";
        if is_resnet {
            if tokens == 0 || tokens > 19 {
                return Err(RuntimeError::Shape("board".to_string()));
            }
            if dim == 0 || dim > 256 {
                return Err(RuntimeError::Shape("channels".to_string()));
            }
        } else if flex {
            if tokens != 6 && tokens != 8 && tokens != 10 {
                return Err(RuntimeError::Shape("tokens".to_string()));
            }
            if dim != 24 && dim != 32 && dim != 40 {
                return Err(RuntimeError::Shape("token dim".to_string()));
            }
        } else if tokens != 8 || dim != 32 {
            return Err(RuntimeError::Shape("tokens".to_string()));
        }
        let vocabs = if game == spec::GAME_SHOGI {
            crate::shogi::SHOGI_VOCABS
        } else if game == spec::GAME_XIANGQI {
            crate::xiangqi::XIANGQI_VOCABS
        } else {
            spec::VOCAB_SIZES
        };
        if is_resnet {
            let tables = Tables::zeros(dim, vocabs);
            let rw = crate::resnet::ResnetWeights::from_arrays(&m.arrays, &m.header.raw)?;
            let acc = Accumulator::new(tokens, dim);
            return Ok(Evaluator { tables, mixer: None, mh: None, heads: Vec::new(), resnet: Some(rw), phase: 1, arch_id: arch, game, tokens, dim, acc, feats: Vec::new(), ctx: Vec::new(), stack: Vec::new() });
        }
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
            let gate = match m.header.raw.get("gate").and_then(|x| x.as_str()) {
                None => rune_kernel::Gate::Clip,
                Some(s) => rune_kernel::Gate::from_str(s).ok_or_else(|| RuntimeError::InvalidState("unknown gate".to_string()))?,
            };
            let alpha = m.header.raw.get("alpha").and_then(|x| x.as_f64()).unwrap_or(1.0) as f32;
            let (dyn_u, dyn_w, ctx_dim) = if arch == "RUNE-REL-02" {
                let cd = m.header.raw.get("context_dim").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                let want_cd = if game == spec::GAME_SHOGI {
                    crate::shogi::SHOGI_CONTEXT_DIM
                } else if game == spec::GAME_XIANGQI {
                    crate::xiangqi::XIANGQI_CONTEXT_DIM
                } else {
                    CONTEXT_DIM
                };
                let pair = (m.arrays.get("dynU"), m.arrays.get("dynW"));
                match pair {
                    (Some(u), Some(w)) => {
                        if cd == 0 || cd != want_cd || u.len() != tokens * cd || w.len() != tokens * cd {
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
            Some(MixerWeights { tokens, dim, wq, bq, wk, bk, wv, bv, gab, dyn_u, dyn_w, ctx_dim, gate, alpha })
        } else {
            None
        };
        let buckets = m.header.raw.get("head_buckets").and_then(|x| x.as_u64()).unwrap_or(1);
        if buckets != 1 && buckets != 3 {
            return Err(RuntimeError::InvalidState("unsupported head_buckets (want 1 or 3)".to_string()));
        }
        let mh = if arch == "RUNE-ATTN-MH4" {
            if tokens != 8 || dim != 32 {
                return Err(RuntimeError::Shape("tokens".to_string()));
            }
            let gate = match m.header.raw.get("gate").and_then(|x| x.as_str()) {
                None => rune_kernel::Gate::Clip,
                Some(s) => rune_kernel::Gate::from_str(s).ok_or_else(|| RuntimeError::InvalidState("unknown gate".to_string()))?,
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
                let q = need_vec(&m.arrays, &format!("wq{}", s))?;
                let b = need_vec(&m.arrays, &format!("bq{}", s))?;
                let k = need_vec(&m.arrays, &format!("wk{}", s))?;
                let kb = need_vec(&m.arrays, &format!("bk{}", s))?;
                let v = need_vec(&m.arrays, &format!("wv{}", s))?;
                let vb = need_vec(&m.arrays, &format!("bv{}", s))?;
                let g = need_vec(&m.arrays, &format!("gab{}", s))?;
                if q.len() != 8 * 32 || b.len() != 8 || k.len() != 8 * 32 || kb.len() != 8 {
                    return Err(RuntimeError::Shape("mh mat".to_string()));
                }
                if v.len() != 8 * 32 || vb.len() != 8 || g.len() != 64 {
                    return Err(RuntimeError::Shape("mh mat".to_string()));
                }
                wq.push(q);
                bq.push(b);
                wk.push(k);
                bk.push(kb);
                wv.push(v);
                bv.push(vb);
                gab.push(g);
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
                gate,
            })
        } else {
            None
        };
        let mut heads: Vec<HeadWeights> = Vec::new();
        for b in 0..buckets {
            let suffix = if buckets == 1 { String::new() } else { format!("_b{}", b) };
            let w1 = need_vec(&m.arrays, &format!("w1{}", suffix))?;
            let b1 = need_vec(&m.arrays, &format!("b1{}", suffix))?;
            let w2 = need_vec(&m.arrays, &format!("w2{}", suffix))?;
            let b2 = need_vec(&m.arrays, &format!("b2{}", suffix))?;
            let wvo = need_vec(&m.arrays, &format!("wvo{}", suffix))
                .or_else(|_| need_vec(&m.arrays, &format!("wv{}", suffix)))?;
            let bvo = scalar_of(&m.arrays, &format!("bvo{}", suffix))
                .or_else(|_| scalar_of(&m.arrays, &format!("bv{}", suffix)))?;
            let wwdl = need_vec(&m.arrays, &format!("wwdl{}", suffix))?;
            let bwdl = need_vec(&m.arrays, &format!("bwdl{}", suffix))?;
            let h1 = b1.len();
            let h2 = b2.len();
            let input = tokens * dim;
            if w1.len() != h1 * input || (w2.len() != h2 * h1 && w2.len() != h2 * h1 * 2) {
                return Err(RuntimeError::Shape("head mat".to_string()));
            }
            if wvo.len() != h2 {
                return Err(RuntimeError::Shape("wvo".to_string()));
            }
            if wwdl.len() != 3 * h2 || bwdl.len() != 3 {
                return Err(RuntimeError::Shape("wdl".to_string()));
            }
            heads.push(HeadWeights { input, h1, h2, w1, b1, w2, b2, wvo, bvo, wwdl, bwdl });
        }
        let acc = Accumulator::new(tokens, dim);
        Ok(Evaluator { tables, mixer, mh, heads, resnet: None, phase: 1, arch_id: arch, game, tokens, dim, acc, feats: Vec::new(), ctx: Vec::new(), stack: Vec::new() })
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
    pub fn refresh(&mut self, board: &Board) {
        if self.game != spec::GAME_CHESS {
            panic!("chess refresh on {} model", self.game);
        }
        let f = extract_features(board);
        self.acc.refresh(&self.tables, &f);
        self.feats = f;
        self.ctx = compute_context(board);
        self.phase = board.game_phase();
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
    pub fn pop(&mut self) {
        let (snap, feats, ctx, phase) = self.stack.pop().expect("pop without push");
        self.acc.restore(&snap);
        self.feats = feats;
        self.ctx = ctx;
        self.phase = phase;
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
