#pragma once

#include <cstddef>

namespace rune {
namespace simd {

inline float clippedRelu(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

inline void matVec(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    float acc = bias ? bias[r] : 0.0f;
    const float* row = mat + static_cast<size_t>(r) * static_cast<size_t>(cols);
    for (int c = 0; c < cols; ++c) acc += row[c] * vec[c];
    out[r] = acc;
  }
}

inline void matVecClipped(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols) {
  matVec(mat, vec, bias, out, rows, cols);
  for (int r = 0; r < rows; ++r) out[r] = clippedRelu(out[r]);
}

inline void matMulTT(const float* a, const float* b, float* out, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int t = 0; t < k; ++t) acc += a[i * k + t] * b[j * k + t];
      out[i * n + j] = acc;
    }
  }
}

inline void matMul(const float* a, const float* b, float* out, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int t = 0; t < k; ++t) acc += a[i * k + t] * b[t * n + j];
      out[i * n + j] = acc;
    }
  }
}

}
}
