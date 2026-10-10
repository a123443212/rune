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

use rune_search::cache::{EvalCache, cache_key};
use rune_search::contract::{canonical_value, engine_score, value_wdl_consistent};
use rune_search::lazy::{LazyConfig, LazyMode, should_refine};

#[test]
fn scale_shared_with_cpp() {
    assert_eq!(canonical_value(2.0), 1.0);
    assert_eq!(canonical_value(-2.0), -1.0);
    assert_eq!(engine_score(1.0), 1000);
    assert_eq!(engine_score(-1.0), -1000);
    assert!(value_wdl_consistent(0.5, &[0.7, 0.2, 0.1], 0.35));
    assert!(!value_wdl_consistent(0.9, &[0.1, 0.1, 0.8], 0.35));
}

#[test]
fn stm_in_cache_key() {
    let a = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let b = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1";
    assert_ne!(cache_key(a, "h", "full"), cache_key(b, "h", "full"));
    assert_ne!(cache_key(a, "hA", "full"), cache_key(a, "hB", "full"));
}

#[test]
fn cache_rejects_foreign_model() {
    let mut c = EvalCache::new("modelA", "full", 4);
    assert!(c.get("f1").is_none());
    c.put("f1", 0.5);
    assert_eq!(c.get("f1"), Some(0.5));
    let mut d = EvalCache::new("modelB", "full", 4);
    assert!(d.get("f1").is_none());
}

#[test]
fn ep_in_cache_key_and_cap_zero_disables() {
    let a = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 2";
    let b = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2";
    assert_ne!(cache_key(a, "h", "full"), cache_key(b, "h", "full"));
    let mut c = EvalCache::new("h", "full", 0);
    for i in 0..100 {
        c.put(&format!("f{}", i), 0.5);
    }
    assert!(c.get("f0").is_none());
    assert!(c.get("f99").is_none());
}

#[test]
fn lazy_modes_match_spec() {
    let l0 = LazyConfig::default();
    let l1 = LazyConfig { mode: LazyMode::L1, ..Default::default() };
    let l2 = LazyConfig { mode: LazyMode::L2, ..Default::default() };
    assert!(should_refine(0.1, 0.0, 0.3, 0.0, 0.5, &l0));
    assert!(!should_refine(0.1, 0.0, 0.3, 0.0, 0.5, &l1));
    assert!(should_refine(0.9, 0.0, 0.3, 0.0, 0.5, &l1));
    assert!(!should_refine(0.9, 0.0, 0.3, 0.0, 0.5, &l2));
    assert!(should_refine(0.15, 0.0, 0.3, 0.9, 0.5, &l2));
}
