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

use sha2::{Digest, Sha256};

pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for b in data {
        h ^= *b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

pub fn fnv_hex(data: &[u8]) -> String {
    format!("{:016x}", fnv1a64(data))
}

pub fn source_hash(header_json: &[u8], payload: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(header_json);
    h.update(payload);
    let d = h.finalize();
    hex_of(&d[..8])
}

pub fn plan_hash(kernel_plan_json: &[u8], memory_json: &[u8], target_json: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(kernel_plan_json);
    h.update(memory_json);
    h.update(target_json);
    let d = h.finalize();
    hex_of(&d[..8])
}

pub fn cache_key(src: &str, arch: &str, quant: &str, isa: &str, compiler: &str) -> String {
    let raw = format!("{}|{}|{}|{}|{}", src, arch, quant, isa, compiler);
    let mut h = Sha256::new();
    h.update(raw.as_bytes());
    let d = h.finalize();
    hex_of(&d[..8])
}

fn hex_of(b: &[u8]) -> String {
    let mut s = String::new();
    for v in b {
        s.push_str(&format!("{:02x}", v));
    }
    s
}

pub fn write_compiled(header: serde_json::Value, payload: &[u8]) -> Vec<u8> {
    let hb = serde_json::to_vec(&header).unwrap_or_default();
    let mut out = Vec::new();
    out.extend_from_slice(b"RUNE");
    let n = hb.len() as u32;
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&hb);
    out.extend_from_slice(payload);
    out
}

pub fn read_header(data: &[u8]) -> Result<(serde_json::Value, usize), String> {
    if data.len() < 8 {
        return Err("truncated".to_string());
    }
    if &data[0..4] != b"RUNE" {
        return Err("bad magic".to_string());
    }
    let n = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    if n > 1000000 {
        return Err("header too large".to_string());
    }
    if data.len() < 8 + n {
        return Err("truncated header".to_string());
    }
    let v: serde_json::Value = serde_json::from_slice(&data[8..8 + n]).map_err(|e| e.to_string())?;
    Ok((v, 8 + n))
}
