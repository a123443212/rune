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
#include <cstdint>
namespace rune {
namespace ref {
float clippedRelu(float x);
float hardSigmoid(float s);
float clampDelta(float v);
int32_t quantizeHalfAway(float w, float scale, int bound);
float dequantize(int q, float scale);
void matVec(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols);
void matVecClipped(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols);
void matMulTT(const float* a, const float* b, float* out, int m, int n, int k);
void matMul(const float* a, const float* b, float* out, int m, int n, int k);
void tokensClip(const float* acc, float* tok, int n);
void tokensDequantClip(const int32_t* acc, float scale, float* tok, int n);
bool routingRefine(float score, float threshold, float tHigh, bool hasTLow, float tLow);
}
}
