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

#include <vector>

#include "core/go/go_scratch.h"

namespace rune {
namespace go {

struct GoResnetSizes {
  int board = 9;
  int channels = 8;
  int blocks = 1;
  int policySize = 82;
  int valueH2 = 32;
  int inPlanes = 1;
};

struct GoResnetWeights {
  std::vector<float> stemW;
  std::vector<float> stemB;
  std::vector<std::vector<float>> blockW1;
  std::vector<std::vector<float>> blockB1;
  std::vector<std::vector<float>> blockW2;
  std::vector<std::vector<float>> blockB2;
  std::vector<float> vh1;
  std::vector<float> bh1;
  std::vector<float> wv;
  float bv = 0.0f;
  std::vector<float> wwdl;
  std::vector<float> bwdl;
  std::vector<float> wpol;
  std::vector<float> bpol;
};

struct GoResnetOutput {
  float value = 0.0f;
  float wdl[3] = {0.0f, 0.0f, 0.0f};
  std::vector<float> policy;
};

GoResnetOutput forwardGoResnet(const GoResnetWeights& wt, const GoResnetSizes& sz,
                               const float* planes);
GoResnetOutput forwardGoResnetFast(const GoResnetWeights& wt, const GoResnetSizes& sz,
                                   const float* planes, GoResnetScratch& sc);

}
}
