#pragma once

#include <cstdint>
#include <vector>

#include "core/features/feature_set.h"
#include "core/go/go_board.h"

namespace rune {
namespace go {

class GoFeatureSet {
 public:
  static constexpr int kNumGroups = 9;
  static constexpr int kTokens = 8;
  static constexpr int kTokenDim = 32;
  static constexpr int kContextDim = 12;

  static int vocabSize(int group);
  static const char* version();

  static void extract(const GoBoard& board, std::vector<ActiveFeature>& out);
  static void context(const GoBoard& board, float* out);
  static int phase(const GoBoard& board);
};

}
}
