#pragma once

#include "core/board/board.h"

namespace rune {

struct ContextSpec {
  static constexpr int kDim = 8;
  static const char* names[8];
};

void computeContext(const Board& board, float* ctx);

}
