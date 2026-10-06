#pragma once
namespace rune {
namespace fused {
void qkvFused8x32(const float* wq, const float* bq, const float* wk, const float* bk, const float* wv, const float* bv, const float* x, float* q, float* k, float* v);
void scoreBiasGate8x8(const float* q, const float* k, const float* gab, bool hard, float* scores, float* gate);
void mixResidual8x32(const float* g, const float* v, const float* x, float alpha, float* tmp, float* out);
void linearBiasClip128(const float* w, const float* b, const float* in, float* out, int rows, int cols);
float dotTanh(const float* wvo, float bvo, const float* h2, int n);
void wdl3x32(const float* w, const float* b, const float* h2, float* wdl);
void matVecClippedFused(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols);
}
}
