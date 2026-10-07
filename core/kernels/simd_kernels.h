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
