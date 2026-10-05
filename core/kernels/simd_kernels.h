#pragma once
#include <cstddef>
namespace rune {
namespace kern {
enum class Path { Scalar, Avx2 };
Path activePath();
const char* activePathName();
bool hasAvx2();
void matVec(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols);
void matVecClipped(const float* mat, const float* vec, const float* bias, float* out, int rows, int cols);
void matMulTT(const float* a, const float* b, float* out, int m, int n, int k);
void matMul(const float* a, const float* b, float* out, int m, int n, int k);
void setPathForTest(Path p);
void clearPathForTest();
}
}
