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

pub mod artifact;
pub mod cache;
pub mod incremental;
pub mod optimizer;
pub mod parser;
pub mod parser_classic;
pub mod parser_common;
pub mod parser_resnet;
pub mod planner;
pub mod planner_fusion;
pub mod planner_memory;
pub mod verifier;

pub use parser::build_from_model;
pub use verifier::verify_bytes;
pub use planner::select_kernels;
pub use optimizer::fuse_plan;
pub use artifact::{write_compiled, read_header, plan_hash, source_hash, cache_key};
pub use cache::{cache_lookup, cache_store};
