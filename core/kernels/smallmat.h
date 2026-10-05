#pragma once
#include <cstddef>
namespace rune {
namespace small {
template <int Rows, int Cols>
inline void matVecFixed(const float* mat, const float* vec, const float* bias, float* out) {
  for (int r = 0; r < Rows; ++r) {
    float acc = bias ? bias[r] : 0.0f;
    const float* row = mat + (size_t)r * Cols;
    int c = 0;
    for (; c + 4 <= Cols; c += 4) {
      acc += row[c] * vec[c] + row[c + 1] * vec[c + 1] + row[c + 2] * vec[c + 2] + row[c + 3] * vec[c + 3];
    }
    for (; c < Cols; ++c) acc += row[c] * vec[c];
    out[r] = acc;
  }
}
template <int Rows, int Cols>
inline void matVecClippedFixed(const float* mat, const float* vec, const float* bias, float* out) {
  matVecFixed<Rows, Cols>(mat, vec, bias, out);
  for (int r = 0; r < Rows; ++r) {
    float v = out[r];
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    out[r] = v;
  }
}
template <int T, int D>
inline void scoreFixed(const float* q, const float* k, float* out) {
  for (int a = 0; a < T; ++a) {
    for (int b = 0; b < T; ++b) {
      float acc = 0.0f;
      for (int t = 0; t < D; ++t) acc += q[a * D + t] * k[b * D + t];
      out[a * T + b] = acc;
    }
  }
}
template <int T, int D>
inline void mixFixed(const float* g, const float* v, float* out) {
  for (int a = 0; a < T; ++a)
    for (int d = 0; d < D; ++d) {
      float acc = 0.0f;
      for (int b = 0; b < T; ++b) acc += g[a * T + b] * v[b * D + d];
      out[a * D + d] = acc;
    }
}
void matVec32(const float* mat, const float* vec, const float* bias, float* out);
void matVec128x256(const float* mat, const float* vec, const float* bias, float* out);
void matVec128x256Clipped(const float* mat, const float* vec, const float* bias, float* out);
void matVec32x128Clipped(const float* mat, const float* vec, const float* bias, float* out);
void score8x8d32(const float* q, const float* k, float* out);
void mix8x8d32(const float* g, const float* v, float* out);
}
}
