#include "core/kernels/fused.h"
#include <cmath>
#include "core/kernels/ref_kernels.h"
#include "core/kernels/smallmat.h"
namespace rune {
namespace fused {
void qkvFused8x32(const float* wq, const float* bq, const float* wk, const float* bk, const float* wv, const float* bv, const float* x, float* q, float* k, float* v) {
  for (int t = 0; t < 8; ++t) {
    small::matVecFixed<32, 32>(wq, x + t * 32, bq, q + t * 32);
    small::matVecFixed<32, 32>(wk, x + t * 32, bk, k + t * 32);
    small::matVecFixed<32, 32>(wv, x + t * 32, bv, v + t * 32);
  }
}
void scoreBiasGate8x8(const float* q, const float* k, const float* gab, bool hard, float* scores, float* gate) {
  small::scoreFixed<8, 32>(q, k, scores);
  for (int i = 0; i < 64; ++i) {
    float b = scores[i] + gab[i];
    scores[i] = b;
    gate[i] = hard ? ref::hardSigmoid(b) : ref::clippedRelu(b);
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
}
}
