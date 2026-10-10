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

pub const RUNTIME_SPEC: &str = "RUNE-12";
pub const FEATURE_VERSION: &str = "grouped_hkav2_fullthreats_v02";
pub const GAME_CHESS: &str = "chess";
pub const GAME_SHOGI: &str = "shogi";
pub const GAME_XIANGQI: &str = "xiangqi";
pub const GAME_GO: &str = "go";

pub fn game_feature_version(game: &str) -> Option<&'static str> {
    match game {
        "chess" => Some(FEATURE_VERSION),
        "shogi" => Some("shogi_raw_v01"),
        "xiangqi" => Some("xiangqi_raw_v01"),
        "go" => Some("go_planes_v01"),
        _ => None,
    }
}

pub fn accepted_feature_versions(game: &str) -> &'static [&'static str] {
    match game {
        "go" => &["go_planes_v01", "go_planes_v02"],
        _ => &[],
    }
}

pub const CONTEXT_DIM: usize = 17;
pub const NUM_GROUPS: usize = 9;
pub const TOKEN_DIM_DEFAULT: usize = 32;
pub const VOCAB_SIZES: [usize; 9] = [256, 256, 256, 128, 128, 512, 512, 64, 4560];
pub const GROUP_NAMES: [&str; 9] = ["pawn_structure", "king_zone", "minor_pieces", "rooks", "queens", "threats", "mobility", "global", "pawn_pairs"];
