#include "core/kernels/fused.h"
#include <cmath>
#include "core/kernels/ref_kernels.h"
#include "core/kernels/simd_kernels.h"
#include "core/kernels/smallmat.h"
#if defined(__AVX2__) || defined(RUNE_X86_INTRIN)
#include <immintrin.h>
#endif
namespace rune {
namespace fused {
void qkvFused8x32(const float* wq, const float* bq, const float* wk, const float* bk, const float* wv, const float* bv, const float* x, float* q, float* k, float* v) {
  for (int t = 0; t < 8; ++t) {
    small::matVecFixed<32, 32>(wq, x + t * 32, bq, q + t * 32);
    small::matVecFixed<32, 32>(wk, x + t * 32, bk, k + t * 32);
    small::matVecFixed<32, 32>(wv, x + t * 32, bv, v + t * 32);
  }
}
void scoreBiasGate8x8(const float* q, const float* k, const float* gab, GateFn gate, float* scores, float* gateOut) {
  small::scoreFixed<8, 32>(q, k, scores);
  for (int i = 0; i < 64; ++i) {
    float b = scores[i] + gab[i];
    scores[i] = b;
    gateOut[i] = applyGate(gate, b);
  }
}
void mixResidual8x32(const float* g, const float* v, const float* x, float alpha, float* tmp, float* out) {
  small::mixFixed<8, 32>(g, v, tmp);
  for (int i = 0; i < 256; ++i) out[i] = x[i] + alpha * tmp[i];
}
void linearBiasClip128(const float* w, const float* b, const float* in, float* out, int rows, int cols) {
  if (rows == 128 && cols == 256) {
    small::matVecClippedFixed<128, 256>(w, in, b, out);
    return;
  }
  if (rows == 32 && cols == 128) {
    small::matVecClippedFixed<32, 128>(w, in, b, out);
    return;
  }
  ref::matVecClipped(w, in, b, out, rows, cols);
}
float dotTanh(const float* wvo, float bvo, const float* h2, int n) {
  float acc = bvo;
  for (int i = 0; i < n; ++i) acc += wvo[i] * h2[i];
  return std::tanh(acc);
}
void wdl3x32(const float* w, const float* b, const float* h2, float* wdl) {
  for (int r = 0; r < 3; ++r) {
    float acc = b[r];
    for (int c = 0; c < 32; ++c) acc += w[r * 32 + c] * h2[c];
    wdl[r] = acc;
  }
}
#if defined(__AVX2__) || defined(RUNE_X86_INTRIN)
RUNE_TARGET_AVX2 void matVecClippedFusedAvx2(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    __m256 acc = _mm256_setzero_ps();
    const float* row = mat + static_cast<size_t>(r) * static_cast<size_t>(cols);
    int c = 0;
    for (; c + 8 <= cols; c += 8) {
      __m256 a = _mm256_loadu_ps(row + c);
      __m256 b = _mm256_loadu_ps(vec + c);
      acc = _mm256_fmadd_ps(a, b, acc);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; c < cols; ++c) s += row[c] * vec[c];
    if (bias) s += bias[r];
    s = s < 0.0f ? 0.0f : (s > 1.0f ? 1.0f : s);
    out[r] = s;
  }
}
#endif
void matVecClippedFused(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols) {
#if defined(__AVX2__) || defined(RUNE_X86_INTRIN)
  if (kern::hasAvx2()) {
    matVecClippedFusedAvx2(mat, vec, bias, out, rows, cols);
    return;
  }
#endif
  for (int r = 0; r < rows; ++r) {
    float acc = bias ? bias[r] : 0.0f;
    const float* row = mat + static_cast<size_t>(r) * static_cast<size_t>(cols);
    for (int c = 0; c < cols; ++c) acc += row[c] * vec[c];
    out[r] = acc < 0.0f ? 0.0f : (acc > 1.0f ? 1.0f : acc);
  }
}
}
}
