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

#include <chrono>
#include <cstdio>
#include <vector>

#include "core/go/go_planes.h"
#include "core/go/go_resnet.h"

using namespace rune::go;

namespace {

double benchFast(const GoResnetWeights& wt, const GoResnetSizes& sz, const float* planes,
                 int iters) {
  GoResnetScratch sc;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    volatile auto r = forwardGoResnetFast(wt, sz, planes, sc);
    (void)r;
  }
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

double benchSlow(const GoResnetWeights& wt, const GoResnetSizes& sz, const float* planes,
                 int iters) {
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    volatile auto r = forwardGoResnet(wt, sz, planes);
    (void)r;
  }
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

void fill(std::vector<float>& v, float s) {
  for (size_t i = 0; i < v.size(); ++i) v[i] = s * (0.5f + (i % 7) * 0.1f);
}

}  // namespace

int main() {
  GoResnetSizes sz;
  sz.board = 9;
  sz.channels = 8;
  sz.blocks = 1;
  sz.policySize = 82;
  sz.valueH2 = 32;
  GoResnetWeights wt;
  wt.stemW.assign(8 * 9, 0.0f);
  wt.stemB.assign(8, 0.0f);
  wt.blockW1.assign(1, std::vector<float>(8 * 8 * 9, 0.0f));
  wt.blockB1.assign(1, std::vector<float>(8, 0.0f));
  wt.blockW2.assign(1, std::vector<float>(8 * 8 * 9, 0.0f));
  wt.blockB2.assign(1, std::vector<float>(8, 0.0f));
  wt.vh1.assign(32 * 8, 0.0f);
  wt.bh1.assign(32, 0.0f);
  wt.wv.assign(32, 0.0f);
  wt.bv = 0.0f;
  wt.wwdl.assign(3 * 32, 0.0f);
  wt.bwdl.assign(3, 0.0f);
  wt.wpol.assign(82 * 8 * 81, 0.0f);
  wt.bpol.assign(82, 0.0f);
  fill(wt.stemW, 0.05f);
  fill(wt.blockW1[0], 0.02f);
  fill(wt.blockW2[0], 0.02f);
  fill(wt.vh1, 0.1f);
  fill(wt.wv, 0.25f);
  fill(wt.wpol, 0.001f);
  std::vector<float> planes(81, 0.0f);
  planes[40] = 1.0f;
  std::printf("slow9x8x1 %.3f ms/eval\n", benchSlow(wt, sz, planes.data(), 20));
  std::printf("fast9x8x1 %.3f ms/eval\n", benchFast(wt, sz, planes.data(), 100));
  GoResnetSizes sz2;
  sz2.board = 19;
  sz2.channels = 16;
  sz2.blocks = 2;
  sz2.policySize = 362;
  sz2.valueH2 = 32;
  GoResnetWeights wt2;
  wt2.stemW.assign(16 * 9, 0.01f);
  wt2.stemB.assign(16, 0.0f);
  wt2.blockW1.assign(2, std::vector<float>(16 * 16 * 9, 0.005f));
  wt2.blockB1.assign(2, std::vector<float>(16, 0.0f));
  wt2.blockW2.assign(2, std::vector<float>(16 * 16 * 9, 0.005f));
  wt2.blockB2.assign(2, std::vector<float>(16, 0.0f));
  wt2.vh1.assign(32 * 16, 0.02f);
  wt2.bh1.assign(32, 0.0f);
  wt2.wv.assign(32, 0.05f);
  wt2.bv = 0.0f;
  wt2.wwdl.assign(3 * 32, 0.02f);
  wt2.bwdl.assign(3, 0.0f);
  wt2.wpol.assign(362 * 16 * 361, 0.0f);
  wt2.bpol.assign(362, 0.0f);
  std::vector<float> planes19(361, 0.0f);
  planes19[180] = 1.0f;
  std::printf("slow19x16x2 %.3f ms/eval\n", benchSlow(wt2, sz2, planes19.data(), 5));
  std::printf("fast19x16x2 %.3f ms/eval\n", benchFast(wt2, sz2, planes19.data(), 10));
  return 0;
}
