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
