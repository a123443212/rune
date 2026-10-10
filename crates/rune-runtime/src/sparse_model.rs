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

use std::collections::HashMap;

use rune_kernel::Gate;
use rune_model::RuneModel;

use crate::accumulator::Tables;
use crate::error::{Result, RuntimeError};
use crate::features::CONTEXT_DIM;
use crate::head_swiglu::SwiGluHead;
use crate::mixer::{HeadWeights, MixerWeights};
use crate::mixer_dual::DualMixer;
use crate::mixer_mh::MultiHeadMixer;
use crate::mixer_soft::{scale_for_dim, SoftMixer};
use crate::model_config::ResolvedModelConfig;

pub(crate) struct SparseModel {
    pub tables: Tables,
    pub mixer: Option<MixerWeights>,
    pub soft: Option<SoftMixer>,
    pub dual: Option<DualMixer>,
    pub multi_head: Option<MultiHeadMixer>,
    pub heads: Vec<HeadWeights>,
}

fn need_vec(arrays: &HashMap<String, Vec<f32>>, name: &str) -> Result<Vec<f32>> {
    arrays
        .get(name)
        .cloned()
        .ok_or_else(|| RuntimeError::TensorMissing(name.to_string()))
}

fn scalar_of(arrays: &HashMap<String, Vec<f32>>, name: &str) -> Result<f32> {
    let values = need_vec(arrays, name)?;
    values
        .first()
        .copied()
        .ok_or_else(|| RuntimeError::Shape(name.to_string()))
}

fn gate_of(model: &RuneModel) -> Result<Gate> {
    match model
        .header
        .raw
        .get("gate")
        .and_then(|value| value.as_str())
    {
        None => Ok(Gate::Clip),
        Some("softmax") => Ok(Gate::Clip),
        Some(name) => Gate::from_str(name)
            .ok_or_else(|| RuntimeError::InvalidState("unknown gate".to_string())),
    }
}

impl SparseModel {
    pub(crate) fn from_model(model: &RuneModel, config: &ResolvedModelConfig) -> Result<Self> {
        if config.is_resnet {
            return Err(RuntimeError::UnsupportedArch(
                config.architecture_id.clone(),
            ));
        }

        let mut tables = Tables::zeros(config.dim, config.vocabs);
        for group in 0..config.vocabs.len() {
            let name = format!("emb{}", group);
            let values = model
                .arrays
                .get(&name)
                .ok_or_else(|| RuntimeError::TensorMissing(name.clone()))?;
            if values.len() != config.vocabs[group] * config.dim {
                return Err(RuntimeError::Shape(name));
            }
            tables.data[group].copy_from_slice(values);
        }

        let has_mixer = matches!(
            config.architecture_id.as_str(),
            "RUNE-ATTN" | "RUNE-ATTN-GAB" | "RUNE-ATTN-SOFT" | "RUNE-REL-02" | "RUNE-REL-LITE" | "RUNE-ATTN-DUAL"
        );
        let mixer = if has_mixer {
            let wq = need_vec(&model.arrays, "wq")?;
            let bq = need_vec(&model.arrays, "bq")?;
            let wk = need_vec(&model.arrays, "wk")?;
            let bk = need_vec(&model.arrays, "bk")?;
            let wv = need_vec(&model.arrays, "wvv").or_else(|_| need_vec(&model.arrays, "wv"))?;
            let bv = need_vec(&model.arrays, "bvv").or_else(|_| need_vec(&model.arrays, "bv"))?;
            if wq.len() != config.dim * config.dim
                || wk.len() != config.dim * config.dim
                || wv.len() != config.dim * config.dim
            {
                return Err(RuntimeError::Shape("mixer mat".to_string()));
            }
            if bq.len() != config.dim || bk.len() != config.dim || bv.len() != config.dim {
                return Err(RuntimeError::Shape("mixer bias".to_string()));
            }
            let gab_name = if model.arrays.contains_key("gabS") {
                "gabS"
            } else {
                "gab"
            };
            let gab = match model.arrays.get(gab_name) {
                Some(values) => values.clone(),
                None if config.architecture_id == "RUNE-ATTN" || config.architecture_id == "RUNE-ATTN-DUAL" => {
                    vec![0.0; config.tokens * config.tokens]
                }
                None => return Err(RuntimeError::TensorMissing(gab_name.to_string())),
            };
            if gab.len() != config.tokens * config.tokens {
                return Err(RuntimeError::Shape("gab".to_string()));
            }

            let (dyn_u, dyn_w, ctx_dim) = if config.architecture_id == "RUNE-REL-02" {
                let context_dim = model
                    .header
                    .raw
                    .get("context_dim")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0) as usize;
                let expected_context_dim = match config.game.as_str() {
                    rune_spec::GAME_SHOGI => crate::shogi::SHOGI_CONTEXT_DIM,
                    rune_spec::GAME_XIANGQI => crate::xiangqi::XIANGQI_CONTEXT_DIM,
                    _ => CONTEXT_DIM,
                };
                match (model.arrays.get("dynU"), model.arrays.get("dynW")) {
                    (Some(u), Some(w)) => {
                        if context_dim == 0
                            || context_dim != expected_context_dim
                            || u.len() != config.tokens * context_dim
                            || w.len() != config.tokens * context_dim
                        {
                            return Err(RuntimeError::Shape("dyn".to_string()));
                        }
                        (u.clone(), w.clone(), context_dim)
                    }
                    (None, None) => (Vec::new(), Vec::new(), 0),
                    _ => return Err(RuntimeError::Shape("dyn pair".to_string())),
                }
            } else {
                (Vec::new(), Vec::new(), 0)
            };

            let gate = gate_of(model)?;
            let alpha = model
                .header
                .raw
                .get("alpha")
                .and_then(|value| value.as_f64())
                .unwrap_or(1.0) as f32;
            Some(MixerWeights {
                tokens: config.tokens,
                dim: config.dim,
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
            })
        } else {
            None
        };

        let (mixer, soft) = if config.architecture_id.as_str() == "RUNE-ATTN-SOFT" {
            match mixer {
                Some(mw) => {
                    let sw = SoftMixer {
                        tokens: mw.tokens,
                        dim: mw.dim,
                        wq: mw.wq,
                        bq: mw.bq,
                        wk: mw.wk,
                        bk: mw.bk,
                        wv: mw.wv,
                        bv: mw.bv,
                        gab: mw.gab,
                        scale: scale_for_dim(mw.dim),
                    };
                    (None, Some(sw))
                }
                None => (None, None),
            }
        } else {
            (mixer, None)
        };

        let dual = if config.architecture_id == "RUNE-ATTN-DUAL" {
            let base = mixer.clone().ok_or_else(|| RuntimeError::Shape("dual base".to_string()))?;
            let wq = need_vec(&model.arrays, "wq2")?;
            let bq = need_vec(&model.arrays, "bq2")?;
            let wk = need_vec(&model.arrays, "wk2")?;
            let bk = need_vec(&model.arrays, "bk2")?;
            let wv = need_vec(&model.arrays, "wvv2").or_else(|_| need_vec(&model.arrays, "wv2"))?;
            let bv = need_vec(&model.arrays, "bvv2").or_else(|_| need_vec(&model.arrays, "bv2"))?;
            if wq.len() != config.dim * config.dim || wk.len() != config.dim * config.dim || wv.len() != config.dim * config.dim {
                return Err(RuntimeError::Shape("dual mat".to_string()));
            }
            if bq.len() != config.dim || bk.len() != config.dim || bv.len() != config.dim {
                return Err(RuntimeError::Shape("dual bias".to_string()));
            }
            let gab2 = match model.arrays.get("gab2") {
                Some(v) => v.clone(),
                None => vec![0.0; config.tokens * config.tokens],
            };
            if gab2.len() != config.tokens * config.tokens {
                return Err(RuntimeError::Shape("gab2".to_string()));
            }
            let second = MixerWeights {
                tokens: config.tokens,
                dim: config.dim,
                wq,
                bq,
                wk,
                bk,
                wv,
                bv,
                gab: gab2,
                dyn_u: Vec::new(),
                dyn_w: Vec::new(),
                ctx_dim: 0,
                gate: base.gate,
                alpha: 1.0,
            };
            Some(DualMixer { first: base, second })
        } else {
            None
        };

        let bucket_count = model
            .header
            .raw
            .get("head_buckets")
            .and_then(|value| value.as_u64())
            .unwrap_or(1);
        if bucket_count != 1 && bucket_count != 3 {
            return Err(RuntimeError::InvalidState(
                "unsupported head_buckets (want 1 or 3)".to_string(),
            ));
        }

        let multi_head = if config.architecture_id == "RUNE-ATTN-MH4" {
            if config.tokens != 8 || config.dim != 32 {
                return Err(RuntimeError::Shape("tokens".to_string()));
            }
            let gate = gate_of(model)?;
            let mut wq = Vec::new();
            let mut bq = Vec::new();
            let mut wk = Vec::new();
            let mut bk = Vec::new();
            let mut wv = Vec::new();
            let mut bv = Vec::new();
            let mut gab = Vec::new();
            for head in 0..4 {
                let suffix = format!("_h{}", head);
                let q = need_vec(&model.arrays, &format!("wq{}", suffix))?;
                let qb = need_vec(&model.arrays, &format!("bq{}", suffix))?;
                let k = need_vec(&model.arrays, &format!("wk{}", suffix))?;
                let kb = need_vec(&model.arrays, &format!("bk{}", suffix))?;
                let v = need_vec(&model.arrays, &format!("wv{}", suffix))?;
                let vb = need_vec(&model.arrays, &format!("bv{}", suffix))?;
                let bias = need_vec(&model.arrays, &format!("gab{}", suffix))?;
                if q.len() != 8 * 32 || qb.len() != 8 || k.len() != 8 * 32 || kb.len() != 8 {
                    return Err(RuntimeError::Shape("mh mat".to_string()));
                }
                if v.len() != 8 * 32 || vb.len() != 8 || bias.len() != 64 {
                    return Err(RuntimeError::Shape("mh mat".to_string()));
                }
                wq.push(q);
                bq.push(qb);
                wk.push(k);
                bk.push(kb);
                wv.push(v);
                bv.push(vb);
                gab.push(bias);
            }
            let wo = need_vec(&model.arrays, "wo")?;
            let bwo = need_vec(&model.arrays, "bwo")?;
            if wo.len() != 32 * 32 || bwo.len() != 32 {
                return Err(RuntimeError::Shape("mh wo".to_string()));
            }
            Some(MultiHeadMixer {
                heads: 4,
                head_dim: 8,
                tokens: config.tokens,
                dim: config.dim,
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

        let mut heads = Vec::new();
        for bucket in 0..bucket_count {
            let suffix = if bucket_count == 1 {
                String::new()
            } else {
                format!("_b{}", bucket)
            };
            let is_swiglu = model.arrays.contains_key(&format!("wgate{}", suffix));
            let w1 = if is_swiglu {
                Vec::new()
            } else {
                need_vec(&model.arrays, &format!("w1{}", suffix))?
            };
            let b1 = if is_swiglu {
                Vec::new()
            } else {
                need_vec(&model.arrays, &format!("b1{}", suffix))?
            };
            let w2 = need_vec(&model.arrays, &format!("w2{}", suffix))?;
            let b2 = need_vec(&model.arrays, &format!("b2{}", suffix))?;
            let wvo = need_vec(&model.arrays, &format!("wvo{}", suffix))
                .or_else(|_| need_vec(&model.arrays, &format!("wv{}", suffix)))?;
            let bvo = scalar_of(&model.arrays, &format!("bvo{}", suffix))
                .or_else(|_| scalar_of(&model.arrays, &format!("bv{}", suffix)))?;
            let wwdl = need_vec(&model.arrays, &format!("wwdl{}", suffix))?;
            let bwdl = need_vec(&model.arrays, &format!("bwdl{}", suffix))?;
            let h2 = b2.len();
            let input = config.tokens * config.dim;
            let swiglu = if is_swiglu {
                let wgate = need_vec(&model.arrays, &format!("wgate{}", suffix))?;
                let bgate = need_vec(&model.arrays, &format!("bgate{}", suffix))?;
                let wup = need_vec(&model.arrays, &format!("wup{}", suffix))?;
                let bup = need_vec(&model.arrays, &format!("bup{}", suffix))?;
                let h1 = bgate.len();
                if bup.len() != h1 {
                    return Err(RuntimeError::Shape("swiglu bias".to_string()));
                }
                if wgate.len() != h1 * input || wup.len() != h1 * input {
                    return Err(RuntimeError::Shape("swiglu mat".to_string()));
                }
                if w2.len() != h2 * h1 {
                    return Err(RuntimeError::Shape("head mat".to_string()));
                }
                Some(SwiGluHead {
                    input,
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
            let h1 = b1.len();
            if !is_swiglu && (w1.len() != h1 * input || (w2.len() != h2 * h1 && w2.len() != h2 * h1 * 2)) {
                return Err(RuntimeError::Shape("head mat".to_string()));
            }
            if wvo.len() != h2 {
                return Err(RuntimeError::Shape("wvo".to_string()));
            }
            if wwdl.len() != 3 * h2 || bwdl.len() != 3 {
                return Err(RuntimeError::Shape("wdl".to_string()));
            }
            heads.push(HeadWeights {
                input,
                h1: if is_swiglu { swiglu.as_ref().map(|s| s.h1).unwrap_or(0) } else { h1 },
                h2,
                w1,
                b1,
                w2,
                b2,
                wvo,
                bvo,
                wwdl,
                bwdl,
                swiglu,
            });
        }

        Ok(Self {
            tables,
            mixer,
            soft,
            dual,
            multi_head,
            heads,
        })
    }
}
