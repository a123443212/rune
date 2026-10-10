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

pub mod activations;
pub mod arena;
pub mod conv;
pub mod conv3x3;
pub mod dense;
pub mod fused;
pub mod policy;
pub mod quant;
pub mod scale;
pub mod simd;
pub mod specialized;
pub mod util;

pub use activations::{clamp_delta, clipped_relu, hard_sigmoid, relu, relu_inplace, screlu, tokens_clip, tokens_dequant_clip, Gate};
pub use conv::{conv2d_nchw, global_avg_pool, residual_add, residual_add_relu, residual_add_relu_inplace};
pub use conv3x3::{conv3x3_pad1, conv3x3_pad1_relu};
pub use dense::{mat_mul, mat_mul_tt, mat_vec, mat_vec_clipped, mat_vec_scalar};
pub use policy::{normalize_policy, policy_entropy, softmax};
pub use scale::{dequantize, quantize_half_away, symmetric_scale};
pub use simd::{active_path, active_path_name, clear_path_for_test, set_path_for_test, uses_simd, KernelPath};
pub use util::{max_abs_diff, routing_refine, spec_check};
