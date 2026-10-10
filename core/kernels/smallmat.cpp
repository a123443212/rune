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

#include "core/kernels/smallmat.h"
namespace rune {
namespace small {
void matVec32(const float* mat, const float* vec, const float* bias, float* out) {
  matVecFixed<32, 32>(mat, vec, bias, out);
}
void matVec128x256(const float* mat, const float* vec, const float* bias, float* out) {
  matVecFixed<128, 256>(mat, vec, bias, out);
}
void matVec128x256Clipped(const float* mat, const float* vec, const float* bias, float* out) {
  matVecClippedFixed<128, 256>(mat, vec, bias, out);
}
void matVec32x128Clipped(const float* mat, const float* vec, const float* bias, float* out) {
  matVecClippedFixed<32, 128>(mat, vec, bias, out);
}
void score8x8d32(const float* q, const float* k, float* out) {
  scoreFixed<8, 32>(q, k, out);
}
void mix8x8d32(const float* g, const float* v, float* out) {
  mixFixed<8, 32>(g, v, out);
}
}
}
