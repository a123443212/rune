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

#include "core/board/board.h"

namespace rune {

struct ActiveFeature {
  uint8_t group = 0;
  uint16_t index = 0;
  bool operator<(const ActiveFeature& o) const {
    if (group != o.group) return group < o.group;
    return index < o.index;
  }
  bool operator==(const ActiveFeature& o) const { return group == o.group && index == o.index; }
};

class GroupedFeatureSet {
 public:
  static constexpr int kNumGroups = 9;
  static constexpr int kTokens = 8;
  static constexpr int kTokenDim = 32;
  static constexpr int kPairVocab = 4560;

  static int vocabSize(int group);
  static const char* version();
  static const char* groupName(int group);
  static int tokenForGroup(int group) { return group == 8 ? 0 : group; }
  static int pairIndex(int ida, int idb);

  static void extract(const Board& board, std::vector<ActiveFeature>& out);
  static void diffFeatures(const std::vector<ActiveFeature>& before, const std::vector<ActiveFeature>& after,
                           std::vector<ActiveFeature>& added, std::vector<ActiveFeature>& removed);
};

}
