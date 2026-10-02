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
  static constexpr int kNumGroups = 8;
  static constexpr int kTokenDim = 32;

  static int vocabSize(int group);
  static const char* version();
  static const char* groupName(int group);

  static void extract(const Board& board, std::vector<ActiveFeature>& out);
  static void diffFeatures(const std::vector<ActiveFeature>& before, const std::vector<ActiveFeature>& after,
                           std::vector<ActiveFeature>& added, std::vector<ActiveFeature>& removed);
};

}
