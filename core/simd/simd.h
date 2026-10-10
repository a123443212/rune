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

#pragma once

#include <cstddef>

#include "core/kernels/ref_kernels.h"
#include "core/kernels/simd_kernels.h"

namespace rune {
namespace simd {

inline float clippedRelu(float x) { return ref::clippedRelu(x); }

inline void matVec(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols) {
  kern::matVec(mat, vec, bias, out, rows, cols);
}

inline void matVecClipped(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols) {
  kern::matVecClipped(mat, vec, bias, out, rows, cols);
}

inline void matMulTT(const float* a, const float* b, float* out, int m, int n, int k) {
  kern::matMulTT(a, b, out, m, n, k);
}

inline void matMul(const float* a, const float* b, float* out, int m, int n, int k) {
  kern::matMul(a, b, out, m, n, k);
}

}
}
