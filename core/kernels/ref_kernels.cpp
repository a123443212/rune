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

#include "core/kernels/ref_kernels.h"
#include <cmath>
#include <cstdint>
namespace rune {
namespace ref {
float clippedRelu(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}
float hardSigmoid(float s) {
  float t = 0.2f * s + 0.5f;
  if (t < 0.0f) return 0.0f;
  if (t > 1.0f) return 1.0f;
  return t;
}
float clampDelta(float v) {
  if (v < -0.25f) return -0.25f;
  if (v > 0.25f) return 0.25f;
  return v;
}
int32_t quantizeHalfAway(float w, float scale, int bound) {
  if (!(scale > 0.0f)) return 0;
  if (!(w == w)) return 0;
  float q = w / scale;
  float r = q >= 0.0f ? static_cast<float>(std::floor(q + 0.5f)) : static_cast<float>(std::ceil(q - 0.5f));
  if (r > static_cast<float>(bound)) r = static_cast<float>(bound);
  if (r < static_cast<float>(-bound)) r = static_cast<float>(-bound);
  if (q != q) return 0;
  if (w > 1e30f) return bound;
  if (w < -1e30f) return -bound;
  return static_cast<int32_t>(r);
}
float dequantize(int q, float scale) {
  return static_cast<float>(q) * scale;
}
void matVec(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    float acc = bias ? bias[r] : 0.0f;
    const float* row = mat + static_cast<size_t>(r) * static_cast<size_t>(cols);
    for (int c = 0; c < cols; ++c) acc += row[c] * vec[c];
    out[r] = acc;
  }
}
void matVecClipped(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols) {
  matVec(mat, vec, bias, out, rows, cols);
  for (int r = 0; r < rows; ++r) out[r] = clippedRelu(out[r]);
}
void matMulTT(const float* a, const float* b, float* out, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int t = 0; t < k; ++t) acc += a[i * k + t] * b[j * k + t];
      out[i * n + j] = acc;
    }
  }
}
void matMul(const float* a, const float* b, float* out, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int t = 0; t < k; ++t) acc += a[i * k + t] * b[t * n + j];
      out[i * n + j] = acc;
    }
  }
}
void tokensClip(const float* acc, float* tok, int n) {
  for (int i = 0; i < n; ++i) tok[i] = clippedRelu(acc[i]);
}
void tokensDequantClip(const int32_t* acc, float scale, float* tok, int n) {
  for (int i = 0; i < n; ++i) tok[i] = clippedRelu(static_cast<float>(acc[i]) * scale);
}
bool routingRefine(float score, float threshold, float tHigh, bool hasTLow, float tLow) {
  if (!(score == score)) return true;
  if (score >= tHigh) return true;
  if (hasTLow && score < tLow) return false;
  return score >= threshold;
}
}
}
