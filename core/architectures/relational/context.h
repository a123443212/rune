#pragma once

#include "core/board/board.h"

namespace rune {

struct ContextSpec {
  static constexpr int kDim = 17;
  static const char* names[17];
};

void computeContext(const Board& board, float* ctx);

}
