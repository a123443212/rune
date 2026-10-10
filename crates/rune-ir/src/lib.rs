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

pub mod arch;
pub mod incremental;
pub mod kernels;
pub mod types;
pub mod verify;
pub mod version;

pub use arch::{arch_family, is_resnet_arch, required_ops_for_arch};
pub use kernels::{kernel_for, shape_key};
pub use types::{BufferPlace, FusionGroup, IrModel, IrOp, IrTarget, IrTensor, KernelEntry, MemoryPlan, RuneIr};
pub use verify::{parse_bytes, valid_isa, verify};
pub use version::{ACCEPTED_IR_VERSIONS, ACCEPTED_SPEC_VERSIONS, CLASSIC_REQUIRED_OPS, COMPILER_VERSION, IR_VERSION, PACKING_VERSION, RESNET_OP_KINDS, RESNET_OPTIONAL_OPS, RESNET_REQUIRED_OPS, SPEC_VERSION};
