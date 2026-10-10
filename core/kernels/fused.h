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
#include "core/architectures/base/architecture.h"
namespace rune {
namespace fused {
void qkvFused8x32(const float* wq, const float* bq, const float* wk, const float* bk, const float* wv, const float* bv, const float* x, float* q, float* k, float* v);
void scoreBiasGate8x8(const float* q, const float* k, const float* gab, GateFn gate, float* scores, float* gateOut);
void mixResidual8x32(const float* g, const float* v, const float* x, float alpha, float* tmp, float* out);
void linearBiasClip128(const float* w, const float* b, const float* in, float* out, int rows, int cols);
float dotTanh(const float* wvo, float bvo, const float* h2, int n);
void wdl3x32(const float* w, const float* b, const float* h2, float* wdl);
void matVecClippedFused(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols);
}
}
