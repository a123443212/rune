/*
RUNE — Relational Unified Neural Evaluator
Copyright (C) 2026 a123443212

SPDX-License-Identifier: MIT OR Apache-2.0

This project is dual-licensed under the MIT License and the
Apache License, Version 2.0. You may choose either license
when using, copying, modifying, or distributing this software.

MIT License: https://opensource.org/license/mit
Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, this
software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
*/

#include "core/model_io/model_error.h"
namespace rune {
const char* modelErrorName(ModelError e) {
  switch (e) {
    case ModelError::Ok: return "ok";
    case ModelError::CannotOpen: return "cannot-open";
    case ModelError::BadMagic: return "bad-magic";
    case ModelError::HeaderTooLarge: return "header-too-large";
    case ModelError::Truncated: return "truncated";
    case ModelError::BadHeader: return "bad-header";
    case ModelError::UnsupportedFormat: return "unsupported-format";
    case ModelError::UnsupportedArch: return "unsupported-architecture";
    case ModelError::UnsupportedQuant: return "unsupported-quantization";
    case ModelError::FeatureMismatch: return "feature-version-mismatch";
    case ModelError::TensorMismatch: return "tensor-mismatch";
    case ModelError::ShapeMismatch: return "shape-mismatch";
    case ModelError::ChecksumMismatch: return "checksum-mismatch";
    case ModelError::Oversized: return "oversized-allocation";
    case ModelError::InvalidDim: return "invalid-dim";
  }
  return "unknown";
}
}
