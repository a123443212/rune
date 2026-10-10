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

pub mod games;
pub mod header;
pub mod hash;
pub mod quant;
pub mod tolerance;

pub use games::{accepted_feature_versions, game_feature_version, CONTEXT_DIM, FEATURE_VERSION, GAME_CHESS, GAME_GO, GAME_SHOGI, GAME_XIANGQI, GROUP_NAMES, NUM_GROUPS, TOKEN_DIM_DEFAULT, VOCAB_SIZES, RUNTIME_SPEC};
pub use header::{is_supported_format, is_supported_quant, quant_bound, MAGIC, MAX_HEADER_LEN, MAX_PAYLOAD_BYTES, MAX_TENSOR_ELEMS, MODEL_FORMAT_VERSION};
pub use hash::{fnv1a64, fnv1a_str};
pub use quant::vocab_size;
