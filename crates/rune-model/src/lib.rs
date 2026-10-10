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
use std::fs;
use std::path::Path;
use rune_spec as spec;
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}
#[derive(Debug, Clone)]
pub struct ModelHeader {
    pub format: u32,
    pub architecture_id: String,
    pub architecture_version: String,
    pub feature_version: String,
    pub game: String,
    pub tokens: usize,
    pub token_dim: usize,
    pub quantization: String,
    pub scales: HashMap<String, f32>,
    pub tensors: Vec<TensorMeta>,
    pub model_hash: String,
    pub raw: serde_json::Value,
}
#[derive(Debug, Clone)]
pub struct RuneModel {
    pub header: ModelHeader,
    pub arrays: HashMap<String, Vec<f32>>,
    pub scales_f32: [f32; 9],
    pub payload_bytes: Vec<u8>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    Io(String),
    BadMagic,
    HeaderTooLarge,
    Truncated,
    BadHeader(String),
    UnsupportedFormat(u32),
    UnsupportedArch(String),
    UnsupportedQuant(String),
    GameMismatch(String),
    FeatureMismatch(String),
    TensorMismatch(String),
    ShapeMismatch(String),
    ChecksumMismatch,
    Oversized(String),
}
impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(s) => write!(f, "io: {}", s),
            LoadError::BadMagic => write!(f, "bad magic"),
            LoadError::HeaderTooLarge => write!(f, "header too large"),
            LoadError::Truncated => write!(f, "truncated payload"),
            LoadError::BadHeader(s) => write!(f, "bad header: {}", s),
            LoadError::UnsupportedFormat(v) => write!(f, "unsupported format {}", v),
            LoadError::UnsupportedArch(s) => write!(f, "unsupported architecture {}", s),
            LoadError::UnsupportedQuant(s) => write!(f, "unsupported quantization {}", s),
            LoadError::GameMismatch(s) => write!(f, "game mismatch {}", s),
            LoadError::FeatureMismatch(s) => write!(f, "feature version mismatch {}", s),
            LoadError::TensorMismatch(s) => write!(f, "tensor mismatch {}", s),
            LoadError::ShapeMismatch(s) => write!(f, "shape mismatch {}", s),
            LoadError::ChecksumMismatch => write!(f, "checksum mismatch"),
            LoadError::Oversized(s) => write!(f, "oversized: {}", s),
        }
    }
}
impl std::error::Error for LoadError {}
fn get_str(v: &serde_json::Value, a: &str, b: &str) -> String {
    if let Some(s) = v.get(a).and_then(|x| x.as_str()) {
        return s.to_string();
    }
    if let Some(s) = v.get(b).and_then(|x| x.as_str()) {
        return s.to_string();
    }
    String::new()
}
fn get_u32(v: &serde_json::Value, key: &str, fb: u32) -> u32 {
    v.get(key).and_then(|x| x.as_u64()).unwrap_or(fb as u64) as u32
}
fn need_usize(v: &serde_json::Value, key: &str) -> Result<usize, LoadError> {
    match v.get(key) {
        None => Err(LoadError::BadHeader(format!("missing {}", key))),
        Some(x) => x.as_u64().map(|n| n as usize).ok_or_else(|| LoadError::BadHeader(format!("bad {}", key))),
    }
}
fn need_scale(scales: &HashMap<String, f32>, name: &str) -> Result<f32, LoadError> {
    match scales.get(name) {
        Some(s) if *s > 0.0 && s.is_finite() => Ok(*s),
        _ => Err(LoadError::BadHeader(format!("bad scale {}", name))),
    }
}
fn zero_hash_values(hstr: &str) -> String {
    let mut out = hstr.to_string();
    for key in ["\"model_hash\":\"", "\"checksum\":\"", "\"header_hash\":\""] {
        let mut start = 0;
        loop {
            let hay = out[start..].to_string();
            let found = hay.find(key);
            let pos = match found {
                Some(p) => start + p,
                None => break,
            };
            let v = pos + key.len();
            if v + 17 <= out.len() && out[v..v + 16].chars().all(|c| c.is_ascii_hexdigit()) && out[v + 16..].chars().next() == Some('"') {
                out.replace_range(v..v + 16, "0000000000000000");
                start = v + 16;
            } else {
                start = v;
            }
            if start >= out.len() {
                break;
            }
        }
    }
    out
}
pub fn load(path: &Path) -> Result<RuneModel, LoadError> {
    let bytes = fs::read(path).map_err(|e| LoadError::Io(e.to_string()))?;
    if bytes.len() < 8 {
        return Err(LoadError::Truncated);
    }
    if bytes[0..4] != spec::MAGIC {
        return Err(LoadError::BadMagic);
    }
    let hlen = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if hlen > spec::MAX_HEADER_LEN {
        return Err(LoadError::HeaderTooLarge);
    }
    if bytes.len() < 8 + hlen {
        return Err(LoadError::Truncated);
    }
    let hstr = std::str::from_utf8(&bytes[8..8 + hlen]).map_err(|_| LoadError::BadHeader("utf8".to_string()))?;
    let hv: serde_json::Value = serde_json::from_str(hstr).map_err(|e| LoadError::BadHeader(e.to_string()))?;
    let fmt = get_u32(&hv, "format", 1);
    if !spec::is_supported_format(fmt) {
        return Err(LoadError::UnsupportedFormat(fmt));
    }
    let arch_id = get_str(&hv, "architecture_id", "arch");
    if arch_id.is_empty() {
        return Err(LoadError::BadHeader("missing architecture id".to_string()));
    }
    let arch_ver = get_str(&hv, "architecture_version", "arch_version");
    let game = hv.get("game").and_then(|x| x.as_str()).unwrap_or(spec::GAME_CHESS).to_string();
    let want_feat = spec::game_feature_version(&game).ok_or_else(|| LoadError::GameMismatch(game.clone()))?;
    let feat = get_str(&hv, "feature_version", "feature_set");
    let accepted = spec::accepted_feature_versions(&game);
    if feat != want_feat && !accepted.contains(&feat.as_str()) {
        return Err(LoadError::FeatureMismatch(feat));
    }
    let quant = hv.get("quantization").and_then(|x| x.as_str()).unwrap_or("fp32").to_string();
    if !spec::is_supported_quant(&quant) {
        return Err(LoadError::UnsupportedQuant(quant));
    }
    let tokens = need_usize(&hv, "tokens")?;
    let token_dim = need_usize(&hv, "token_dim")?;
    if tokens == 0 || tokens > 19 || token_dim == 0 || token_dim > 128 {
        return Err(LoadError::ShapeMismatch("tokens or dim out of range".to_string()));
    }
    let tlist = hv.get("tensor_metadata").or_else(|| hv.get("tensors")).ok_or_else(|| LoadError::BadHeader("missing tensors".to_string()))?;
    let arr = tlist.as_array().ok_or_else(|| LoadError::BadHeader("tensors not array".to_string()))?;
    let mut metas: Vec<TensorMeta> = Vec::new();
    for t in arr {
        let name = t.get("name").and_then(|x| x.as_str()).ok_or_else(|| LoadError::BadHeader("tensor name".to_string()))?.to_string();
        let shape_v = t.get("shape").and_then(|x| x.as_array()).ok_or_else(|| LoadError::BadHeader("tensor shape".to_string()))?;
        let mut shape: Vec<usize> = Vec::new();
        for d in shape_v {
            let v = d.as_u64().ok_or_else(|| LoadError::BadHeader("shape dim".to_string()))? as usize;
            if v > 1000000 {
                return Err(LoadError::Oversized("dim too large".to_string()));
            }
            shape.push(v);
        }
        let dtype = t.get("dtype").and_then(|x| x.as_str()).unwrap_or("float32").to_string();
        if dtype != "float32" && dtype != "int8" && dtype != "int16" {
            return Err(LoadError::UnsupportedQuant(dtype));
        }
        metas.push(TensorMeta { name, shape, dtype });
    }
    if metas.is_empty() {
        return Err(LoadError::BadHeader("empty tensor list".to_string()));
    }
    let mut legacy_scales: HashMap<String, f64> = HashMap::new();
    if let Some(sv) = hv.get("scales").and_then(|x| x.as_object()) {
        for (k, v) in sv {
            if let Some(f) = v.as_f64() {
                legacy_scales.insert(k.clone(), f);
            }
        }
    }
    let mut qm_scales: HashMap<String, f64> = HashMap::new();
    if let Some(qm) = hv.get("quantization_metadata").and_then(|x| x.as_object()) {
        if let Some(s2) = qm.get("scales").and_then(|x| x.as_object()) {
            for (k, v) in s2 {
                if let Some(f) = v.as_f64() {
                    qm_scales.insert(k.clone(), f);
                }
            }
        }
    }
    if !legacy_scales.is_empty() && !qm_scales.is_empty() {
        if legacy_scales.len() != qm_scales.len() {
            return Err(LoadError::BadHeader("scale mismatch".to_string()));
        }
        for (k, v) in &legacy_scales {
            match qm_scales.get(k) {
                Some(w) if w == v => {}
                _ => return Err(LoadError::BadHeader("scale mismatch".to_string())),
            }
        }
    }
    let mut scales: HashMap<String, f32> = HashMap::new();
    for (k, v) in &qm_scales {
        scales.insert(k.clone(), *v as f32);
    }
    for (k, v) in &legacy_scales {
        if !scales.contains_key(k) {
            scales.insert(k.clone(), *v as f32);
        }
    }
    let hash_str = get_str(&hv, "model_hash", "checksum");
    let header_hash_str = hv.get("header_hash").and_then(|x| x.as_str()).unwrap_or("").to_string();
    let payload = bytes[8 + hlen..].to_vec();
    if payload.len() > spec::MAX_PAYLOAD_BYTES {
        return Err(LoadError::Oversized("payload too large".to_string()));
    }
    let need_hash = fmt == 2 || arch_id.starts_with("RUNE-03-") || arch_id == "RUNE-04" || arch_id == "RUNE-05";
    if need_hash && hash_str.is_empty() {
        return Err(LoadError::BadHeader("missing model hash".to_string()));
    }
    if !hash_str.is_empty() {
        let got = spec::fnv1a64(&payload);
        let want = u64::from_str_radix(hash_str.trim(), 16).map_err(|_| LoadError::BadHeader("bad hash hex".to_string()))?;
        if got != want {
            return Err(LoadError::ChecksumMismatch);
        }
    }
    if !header_hash_str.is_empty() {
        let zeroed = zero_hash_values(hstr);
        let mut combined = zeroed.into_bytes();
        combined.extend_from_slice(&payload);
        let got = spec::fnv1a64(&combined);
        let want = u64::from_str_radix(header_hash_str.trim(), 16).map_err(|_| LoadError::BadHeader("bad hash hex".to_string()))?;
        if got != want {
            return Err(LoadError::ChecksumMismatch);
        }
    }
    let mut off: usize = 0;
    let mut arrays: HashMap<String, Vec<f32>> = HashMap::new();
    for m in &metas {
        if arrays.contains_key(&m.name) {
            return Err(LoadError::BadHeader(format!("duplicate tensor {}", m.name)));
        }
        let mut count: usize = 1;
        for d in &m.shape {
            count = count.checked_mul(*d).ok_or_else(|| LoadError::Oversized("tensor mul overflow".to_string()))?;
        }
        if count == 0 || count > spec::MAX_TENSOR_ELEMS {
            return Err(LoadError::Oversized("tensor elems out of range".to_string()));
        }
        if m.dtype == "int8" {
            if off + count > payload.len() {
                return Err(LoadError::Truncated);
            }
            let scale = need_scale(&scales, &m.name)?;
            let mut v: Vec<f32> = Vec::with_capacity(count);
            for i in 0..count {
                let q = payload[off + i] as i8 as i32;
                v.push(q as f32 * scale);
            }
            off += count;
            arrays.insert(m.name.clone(), v);
        } else if m.dtype == "int16" {
            if off + count * 2 > payload.len() {
                return Err(LoadError::Truncated);
            }
            let scale = need_scale(&scales, &m.name)?;
            let mut v: Vec<f32> = Vec::with_capacity(count);
            for i in 0..count {
                let lo = payload[off + i * 2] as u16;
                let hi = payload[off + i * 2 + 1] as u16;
                let u = lo | (hi << 8);
                let q = u as i16 as i32;
                v.push(q as f32 * scale);
            }
            off += count * 2;
            arrays.insert(m.name.clone(), v);
        } else {
            if off + count * 4 > payload.len() {
                return Err(LoadError::Truncated);
            }
            let mut v: Vec<f32> = Vec::with_capacity(count);
            for i in 0..count {
                let b = [payload[off + i * 4], payload[off + i * 4 + 1], payload[off + i * 4 + 2], payload[off + i * 4 + 3]];
                v.push(f32::from_le_bytes(b));
            }
            off += count * 4;
            arrays.insert(m.name.clone(), v);
        }
    }
    if off != payload.len() {
        return Err(LoadError::BadHeader("trailing payload bytes".to_string()));
    }
    let mut sarr = [1.0_f32; 9];
    for g in 0..9 {
        let k = format!("emb{}", g);
        if let Some(s) = scales.get(&k) {
            sarr[g] = *s;
        }
    }
    let header = ModelHeader {
        format: fmt,
        architecture_id: arch_id,
        architecture_version: arch_ver,
        feature_version: feat,
        game,
        tokens,
        token_dim,
        quantization: quant,
        scales,
        tensors: metas,
        model_hash: hash_str,
        raw: hv,
    };
    Ok(RuneModel { header, arrays, scales_f32: sarr, payload_bytes: payload })
}
pub fn model_hash_hex(payload: &[u8]) -> String {
    format!("{:016x}", spec::fnv1a64(payload))
}
