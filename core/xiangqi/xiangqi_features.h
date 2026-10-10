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
#include <vector>

#include "core/features/feature_set.h"
#include "core/xiangqi/xiangqi_board.h"

namespace rune {
namespace xiangqi {

class XiangqiFeatureSet {
 public:
  static constexpr int kNumGroups = 9;
  static constexpr int kTokens = 8;
  static constexpr int kTokenDim = 32;
  static constexpr int kContextDim = 12;

  static int vocabSize(int group);
  static const char* version();
  static int tokenForGroup(int group) { return group == 8 ? 0 : group; }

  static void extract(const XiangqiBoard& board, std::vector<ActiveFeature>& out);
  static void context(const XiangqiBoard& board, float* out);
  static int phase(const XiangqiBoard& board);
};

}
}
