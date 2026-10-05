#pragma once

#include <cstddef>

#include "core/kernels/ref_kernels.h"
#include "core/kernels/simd_kernels.h"

namespace rune {
namespace simd {

inline float clippedRelu(float x) { return ref::clippedRelu(x); }

inline void matVec(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols) {
  kern::matVec(mat, vec, bias, out, rows, cols);
}

inline void matVecClipped(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols) {
  kern::matVecClipped(mat, vec, bias, out, rows, cols);
}

inline void matMulTT(const float* a, const float* b, float* out, int m, int n, int k) {
  kern::matMulTT(a, b, out, m, n, k);
}

inline void matMul(const float* a, const float* b, float* out, int m, int n, int k) {
  kern::matMul(a, b, out, m, n, k);
}

}
}
