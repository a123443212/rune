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

#include "core/incremental/incremental_mixer.h"

#include "core/simd/simd.h"

namespace rune {
namespace v13 {

void denseBaselineForward(const IncrWeights& w, const float* tokens, const float* ctx,
                          float* out) {
  RelationalCache c;
  std::string err;
  IncrWeights wc = w;
  if (!c.configure(wc, 1 << 30, err)) return;
  c.rebuild(tokens, ctx);
  const auto& o = c.out();
  size_t n = o.size();
  for (size_t i = 0; i < n; ++i) out[i] = o[i];
}

void qkvRowsForward(const IncrWeights& w, const float* tokens,
                    const std::vector<int>& changed, float* q, float* k, float* vv) {
  int d = w.dim;
  for (int c : changed) {
    simd::matVec(w.wq, tokens + c * d, w.bq, q + c * d, d, d);
    simd::matVec(w.wk, tokens + c * d, w.bk, k + c * d, d, d);
    simd::matVec(w.wv, tokens + c * d, w.bv, vv + c * d, d, d);
  }
}

int scoreCellsForChanged(int tokens, const std::vector<int>& changed) {
  int k = static_cast<int>(changed.size());
  return 2 * k * tokens - k * k;
}

}
}
