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

#include <cmath>
#include <cstddef>
#include <vector>

#include "core/architectures/base/architecture.h"
#include "core/simd/simd.h"

namespace rune {

inline float swigluSilu(float x) { return x / (1.0f + std::exp(-x)); }

inline bool headIsSwiGlu(const HeadBucket& h) { return !h.wgate.empty(); }

inline void swigluForward(const HeadBucket& h, const float* flat, int flatN, float& value, float* wdl,
                          std::vector<float>& scratch) {
  int h1n = static_cast<int>(h.bgate.size());
  int h2n = static_cast<int>(h.b2.size());
  size_t need = static_cast<size_t>(h1n) * 3 + static_cast<size_t>(h2n);
  if (scratch.size() < need) scratch.resize(need);
  float* gate = scratch.data();
  float* up = scratch.data() + h1n;
  float* act = scratch.data() + h1n * 2;
  float* h2 = scratch.data() + h1n * 3;
  simd::matVec(h.wgate.data(), flat, h.bgate.data(), gate, h1n, flatN);
  simd::matVec(h.wup.data(), flat, h.bup.data(), up, h1n, flatN);
  for (int i = 0; i < h1n; ++i) act[i] = swigluSilu(gate[i]) * up[i];
  simd::matVecClipped(h.w2.data(), act, h.b2.data(), h2, h2n, h1n);
  float v = h.bvo[0];
  for (int i = 0; i < h2n; ++i) v += h.wvo[i] * h2[i];
  value = std::tanh(v);
  simd::matVec(h.wwdl.data(), h2, h.bwdl.data(), wdl, 3, h2n);
}

}
