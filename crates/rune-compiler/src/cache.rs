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

use std::fs;
use std::path::{Path, PathBuf};

pub fn cache_file(cache_dir: &Path, key: &str) -> PathBuf {
    cache_dir.join(format!("rune-{}.json", key))
}

pub fn cache_lookup(cache_dir: &Path, key: &str) -> Option<serde_json::Value> {
    let p = cache_file(cache_dir, key);
    let data = fs::read(p).ok()?;
    serde_json::from_slice(&data).ok()
}

pub fn cache_store(cache_dir: &Path, key: &str, doc: &serde_json::Value) -> Result<PathBuf, String> {
    fs::create_dir_all(cache_dir).map_err(|e| e.to_string())?;
    let p = cache_file(cache_dir, key);
    let data = serde_json::to_vec_pretty(doc).map_err(|e| e.to_string())?;
    fs::write(&p, data).map_err(|e| e.to_string())?;
    Ok(p)
}
