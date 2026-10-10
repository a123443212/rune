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

#pragma once
#include <cstddef>
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#define RUNE_X86_INTRIN 1
#define RUNE_TARGET_AVX2 __attribute__((target("avx2,fma")))
#define RUNE_TARGET_AVX512 __attribute__((target("avx512f")))
#else
#define RUNE_TARGET_AVX2
#define RUNE_TARGET_AVX512
#endif
namespace rune {
namespace kern {
enum class Path { Scalar, Avx2, Avx512 };
Path activePath();
const char* activePathName();
bool hasAvx2();
bool hasAvx512();
void matVec(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols);
void matVecClipped(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols);
void matMulTT(const float* a, const float* b, float* out, int m, int n, int k);
void matMul(const float* a, const float* b, float* out, int m, int n, int k);
void setPathForTest(Path p);
void clearPathForTest();
}
}
