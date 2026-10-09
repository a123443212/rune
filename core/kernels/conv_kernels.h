#pragma once

#include <cstddef>
#include <vector>

namespace rune {
namespace conv {

void conv2dNchw(const float* input, const float* weight, const float* bias, float* out, int n,
                int cin, int cout, int h, int w, int kh, int kw, int padH, int padW);
void conv3x3Pad1(const float* input, const float* weight, const float* bias, float* out, int cin,
                 int cout, int h, int w);
void conv3x3Pad1Relu(const float* input, const float* weight, const float* bias, float* out,
                     int cin, int cout, int h, int w);
void reluInplace(float* buf, size_t n);
void residualAddReluInplace(const float* base, float* delta, size_t n);
void residualAddRelu(const float* a, const float* b, float* out, size_t n);
void globalAvgPool(const float* input, float* out, int n, int c, int h, int w);

}
}
