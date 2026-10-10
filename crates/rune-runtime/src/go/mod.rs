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

pub mod board;
pub mod planes;
pub mod rules;
pub mod state;

pub use board::{GoBoard, clamp01_f32, group_and_liberties, liberty_map, neighbors_of, valid_size};
pub use planes::{FEATURE_VERSION_V02, GO_CONTEXT_DIM, GO_VOCABS, PLANES_V02, extract_planes_v02, phase_from_group7, token_features_v02};
pub use rules::{apply_move, area_score, legal_moves, play_stone, territory_owner};
pub use state::{GoState, game_phase, komi_bucket, move_bucket};

pub fn extract_planes(b: &GoBoard) -> Vec<f32> {
    let n = b.size;
    let mut out = vec![0.0f32; n * n];
    for r in 0..n {
        for c in 0..n {
            let v = b.at(r, c);
            let rel: f32 = if v == 0 { 0.0 } else if (v == 1 && b.stm == 0) || (v == -1 && b.stm == 1) { 1.0 } else { -1.0 };
            out[r * n + c] = rel;
        }
    }
    out
}

pub fn compute_context(b: &GoBoard) -> Vec<f32> {
    let mut us = 0usize;
    let mut them = 0usize;
    for v in b.stones.iter() {
        if *v == 0 {
            continue;
        }
        let mine = (*v == 1 && b.stm == 0) || (*v == -1 && b.stm == 1);
        if mine {
            us += 1;
        } else {
            them += 1;
        }
    }
    let total = (b.size * b.size) as f32;
    vec![b.stm as f32, (us as f32 / total).clamp(0.0, 1.0), (them as f32 / total).clamp(0.0, 1.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

pub fn extract_features_v02(state: &str) -> crate::error::Result<Vec<(u8, u16)>> {
    let st = GoState::parse(state)?;
    Ok(token_features_v02(&st))
}

pub fn compute_context_v02(state: &str) -> crate::error::Result<Vec<f32>> {
    let st = GoState::parse(state)?;
    Ok(st.context_v02())
}

pub fn legal_moves_state(state: &str) -> crate::error::Result<Vec<i32>> {
    let st = GoState::parse(state)?;
    let stone_stm = st.board.stm;
    Ok(legal_moves(&st.board.stones, st.board.size, stone_stm, st.ko))
}

pub fn score_state(state: &str) -> crate::error::Result<f32> {
    let st = GoState::parse(state)?;
    Ok(area_score(&st.board.stones, st.board.size, st.komi))
}
