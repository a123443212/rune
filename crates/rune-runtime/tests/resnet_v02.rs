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

use std::path::PathBuf;

#[test]
fn resnet_v02_fixture_loads() {
    let p = PathBuf::from("../../spec/test-vectors/models/resnet9-8plane-fp32.rune");
    let m = rune_model::load(&p).expect("v02 fixture");
    assert_eq!(m.header.architecture_id, "RUNE-RESNET-01");
    assert_eq!(m.header.game, "go");
    assert_eq!(m.header.raw.get("in_planes").and_then(|v| v.as_u64()), Some(8));
    let ev = rune_runtime::evaluator::Evaluator::from_model(&m).expect("evaluator");
    assert!(ev.is_resnet());
}

#[test]
fn resnet_v02_matches_golden() {
    let raw = std::fs::read_to_string("../../spec/test-vectors/resnet/eval_v02.json").expect("read golden");
    let g: serde_json::Value = serde_json::from_str(&raw).expect("parse");
    let p = PathBuf::from("../../spec/test-vectors/models/resnet9-8plane-fp32.rune");
    let m = rune_model::load(&p).expect("fixture");
    let ev = rune_runtime::evaluator::Evaluator::from_model(&m).expect("evaluator");
    let vecs = g["vectors"].as_array().unwrap();
    assert_eq!(vecs.len(), 4);
    for v in vecs {
        let state = v["state"].as_str().unwrap();
        let st = rune_runtime::go::GoState::parse(state).expect("parse golden");
        assert_eq!(st.board.size, 9);
        let planes = rune_runtime::go::extract_planes_v02(&st);
        assert_eq!(planes.len(), 8 * 81);
        let want_shape = v["planes_shape"].as_array().unwrap();
        assert_eq!(want_shape[1].as_u64(), Some(8));
        let legal = rune_runtime::go::legal_moves(&st.board.stones, 9, st.board.stm, st.ko);
        assert_eq!(legal.len() as u64, v["legal_count"].as_u64().unwrap());
        assert!(legal.contains(&-1));
        let r = ev.evaluate_planes(&planes);
        let want = v["value"].as_f64().unwrap() as f32;
        assert!((r.value - want).abs() < 1e-4, "{} vs {} for {}", r.value, want, state);
        let s: f32 = r.policy.iter().sum();
        assert!((s - 1.0).abs() < 1e-4);
    }
}
