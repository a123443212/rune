#include "core/kernels/simd_kernels.h"
#include "core/kernels/ref_kernels.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif
namespace rune {
namespace kern {
namespace {
int gForce = 0;
Path gForced = Path::Scalar;
}
bool hasAvx2() {
#if defined(__AVX2__)
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_cpu_supports("avx2");
#else
  return true;
#endif
#else
  return false;
#endif
}
Path activePath() {
  if (gForce == 1) return gForced;
  if (hasAvx2()) return Path::Avx2;
  return Path::Scalar;
}
const char* activePathName() {
  return activePath() == Path::Avx2 ? "avx2" : "scalar";
}
void setPathForTest(Path p) {
  gForce = 1;
  gForced = p;
}
void clearPathForTest() { gForce = 0; }
#if defined(__AVX2__)
void matVecAvx2(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols) {
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
    out[r] = s;
  }
}
#endif
void matVec(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols) {
  if (activePath() == Path::Avx2) {
#if defined(__AVX2__)
    matVecAvx2(mat, vec, bias, out, rows, cols);
    return;
#endif
  }
  ref::matVec(mat, vec, bias, out, rows, cols);
}
void matVecClipped(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols) {
  matVec(mat, vec, bias, out, rows, cols);
  for (int r = 0; r < rows; ++r) out[r] = ref::clippedRelu(out[r]);
}
void matMulTT(const float* a, const float* b, float* out, int m, int n, int k) {
  ref::matMulTT(a, b, out, m, n, k);
}
void matMul(const float* a, const float* b, float* out, int m, int n, int k) {
  ref::matMul(a, b, out, m, n, k);
}
}
}
