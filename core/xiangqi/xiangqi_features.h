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
