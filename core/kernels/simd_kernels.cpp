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

#include "core/kernels/simd_kernels.h"
#include "core/kernels/ref_kernels.h"
#if defined(__AVX2__) || defined(RUNE_X86_INTRIN)
#include <immintrin.h>
#endif
#if defined(__AVX512F__) || defined(RUNE_X86_INTRIN)
#include <immintrin.h>
#endif
namespace rune {
namespace kern {
namespace {
int gForce = 0;
Path gForced = Path::Scalar;
}
bool hasAvx2() {
#if defined(__AVX2__) || defined(RUNE_X86_INTRIN)
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_cpu_supports("avx2");
#else
  return true;
#endif
#else
  return false;
#endif
}
bool hasAvx512() {
#if defined(__AVX512F__) || defined(RUNE_X86_INTRIN)
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_cpu_supports("avx512f");
#else
  return true;
#endif
#else
  return false;
#endif
}
Path activePath() {
  if (gForce == 1) return gForced;
  if (hasAvx512()) return Path::Avx512;
  if (hasAvx2()) return Path::Avx2;
  return Path::Scalar;
}
const char* activePathName() {
  Path p = activePath();
  if (p == Path::Avx512) return "avx512";
  if (p == Path::Avx2) return "avx2";
  return "scalar";
}
void setPathForTest(Path p) {
  gForce = 1;
  gForced = p;
}
void clearPathForTest() { gForce = 0; }
#if defined(__AVX512F__) || defined(RUNE_X86_INTRIN)
RUNE_TARGET_AVX512 void matVecAvx512(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    __m512 acc = _mm512_setzero_ps();
    const float* row = mat + static_cast<size_t>(r) * static_cast<size_t>(cols);
    int c = 0;
    for (; c + 16 <= cols; c += 16) {
      __m512 a = _mm512_loadu_ps(row + c);
      __m512 b = _mm512_loadu_ps(vec + c);
      acc = _mm512_fmadd_ps(a, b, acc);
    }
    float s = _mm512_reduce_add_ps(acc);
    for (; c < cols; ++c) s += row[c] * vec[c];
    if (bias) s += bias[r];
    out[r] = s;
  }
}
#endif
#if defined(__AVX2__) || defined(RUNE_X86_INTRIN)
RUNE_TARGET_AVX2 void matVecAvx2(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols) {
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
  Path p = activePath();
  if (p == Path::Avx512) {
#if defined(__AVX512F__) || defined(RUNE_X86_INTRIN)
    matVecAvx512(mat, vec, bias, out, rows, cols);
    return;
#endif
  }
  if (p == Path::Avx2) {
#if defined(__AVX2__) || defined(RUNE_X86_INTRIN)
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
#if defined(__AVX512F__) || defined(RUNE_X86_INTRIN)
RUNE_TARGET_AVX512 void matMulTTAvx512(const float* a, const float* b, float* out, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      __m512 acc = _mm512_setzero_ps();
      int t = 0;
      for (; t + 16 <= k; t += 16) {
        __m512 aa = _mm512_loadu_ps(a + i * k + t);
        __m512 bb = _mm512_loadu_ps(b + j * k + t);
        acc = _mm512_fmadd_ps(aa, bb, acc);
      }
      float s = _mm512_reduce_add_ps(acc);
      for (; t < k; ++t) s += a[i * k + t] * b[j * k + t];
      out[i * n + j] = s;
    }
  }
}
RUNE_TARGET_AVX512 void matMulAvx512(const float* a, const float* b, float* out, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) out[i * n + j] = 0.0f;
    for (int t = 0; t < k; ++t) {
      __m512 aa = _mm512_set1_ps(a[i * k + t]);
      int j = 0;
      for (; j + 16 <= n; j += 16) {
        __m512 bb = _mm512_loadu_ps(b + t * n + j);
        __m512 cc = _mm512_loadu_ps(out + i * n + j);
        cc = _mm512_fmadd_ps(aa, bb, cc);
        _mm512_storeu_ps(out + i * n + j, cc);
      }
      for (; j < n; ++j) out[i * n + j] += a[i * k + t] * b[t * n + j];
    }
  }
}
#endif
#if defined(__AVX2__) || defined(RUNE_X86_INTRIN)
RUNE_TARGET_AVX2 void matMulTTAvx2(const float* a, const float* b, float* out, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      __m256 acc = _mm256_setzero_ps();
      int t = 0;
      for (; t + 8 <= k; t += 8) {
        __m256 aa = _mm256_loadu_ps(a + i * k + t);
        __m256 bb = _mm256_loadu_ps(b + j * k + t);
        acc = _mm256_fmadd_ps(aa, bb, acc);
      }
      float tmp[8];
      _mm256_storeu_ps(tmp, acc);
      float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
      for (; t < k; ++t) s += a[i * k + t] * b[j * k + t];
      out[i * n + j] = s;
    }
  }
}
RUNE_TARGET_AVX2 void matMulAvx2(const float* a, const float* b, float* out, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) out[i * n + j] = 0.0f;
    for (int t = 0; t < k; ++t) {
      __m256 aa = _mm256_set1_ps(a[i * k + t]);
      int j = 0;
      for (; j + 8 <= n; j += 8) {
        __m256 bb = _mm256_loadu_ps(b + t * n + j);
        __m256 cc = _mm256_loadu_ps(out + i * n + j);
        cc = _mm256_fmadd_ps(aa, bb, cc);
        _mm256_storeu_ps(out + i * n + j, cc);
      }
      for (; j < n; ++j) out[i * n + j] += a[i * k + t] * b[t * n + j];
    }
  }
}
#endif
void matMulTT(const float* a, const float* b, float* out, int m, int n, int k) {
  Path p = activePath();
  if (p == Path::Avx512) {
#if defined(__AVX512F__) || defined(RUNE_X86_INTRIN)
    matMulTTAvx512(a, b, out, m, n, k);
    return;
#endif
  }
  if (p == Path::Avx2) {
#if defined(__AVX2__) || defined(RUNE_X86_INTRIN)
    matMulTTAvx2(a, b, out, m, n, k);
    return;
#endif
  }
  ref::matMulTT(a, b, out, m, n, k);
}
void matMul(const float* a, const float* b, float* out, int m, int n, int k) {
  Path p = activePath();
  if (p == Path::Avx512) {
#if defined(__AVX512F__) || defined(RUNE_X86_INTRIN)
    matMulAvx512(a, b, out, m, n, k);
    return;
#endif
  }
  if (p == Path::Avx2) {
#if defined(__AVX2__) || defined(RUNE_X86_INTRIN)
    matMulAvx2(a, b, out, m, n, k);
    return;
#endif
  }
  ref::matMul(a, b, out, m, n, k);
}
}
}
