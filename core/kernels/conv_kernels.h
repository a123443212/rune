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
