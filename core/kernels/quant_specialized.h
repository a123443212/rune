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
#include <cstdint>
namespace rune {
namespace qspec {
void dequantClip(const int32_t* acc, float scale, float* out, int n);
void dequantClipBatched(const float* scales8, const int32_t* acc, float* out, int dim);
void clampRow(float* out, int n, float lo, float hi);
int32_t quantizeHalfAwaySpec(float w, float scale, int bound);
}
}
