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

fn check_board(n: usize, blocks: usize, golden: &str, fixture: &str) {
    let p = PathBuf::from(format!("../../spec/test-vectors/models/{}", fixture));
    let m = rune_model::load(&p).expect("big fixture");
    assert_eq!(m.header.architecture_id, "RUNE-RESNET-01");
    assert_eq!(m.header.raw.get("board_size").and_then(|v| v.as_u64()), Some(n as u64));
    assert_eq!(m.header.raw.get("in_planes").and_then(|v| v.as_u64()), Some(8));
    let ev = rune_runtime::evaluator::Evaluator::from_model(&m).expect("evaluator");
    let raw = std::fs::read_to_string(format!("../../spec/test-vectors/resnet/{}", golden)).expect("read golden");
    let g: serde_json::Value = serde_json::from_str(&raw).expect("parse");
    assert_eq!(g["board"].as_u64(), Some(n as u64));
    let _ = blocks;
    for v in g["vectors"].as_array().unwrap() {
        let state = v["state"].as_str().unwrap();
        let st = rune_runtime::go::GoState::parse(state).expect("parse golden");
        assert_eq!(st.board.size, n);
        let planes = rune_runtime::go::extract_planes_v02(&st);
        assert_eq!(planes.len(), 8 * n * n);
        let legal = rune_runtime::go::legal_moves(&st.board.stones, n, st.board.stm, st.ko);
        assert_eq!(legal.len() as u64, v["legal_count"].as_u64().unwrap());
        let r = ev.evaluate_planes(&planes);
        let want = v["value"].as_f64().unwrap() as f32;
        assert!((r.value - want).abs() < 1e-4, "{} vs {} for {}", r.value, want, state);
    }
}

#[test]
fn resnet_v02_13_matches_golden() {
    check_board(13, 1, "eval_v02_13.json", "resnet13-8plane-fp32.rune");
}

#[test]
fn resnet_v02_19_matches_golden() {
    check_board(19, 1, "eval_v02_19.json", "resnet19-8plane-fp32.rune");
}
