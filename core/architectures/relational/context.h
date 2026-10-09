#pragma once

#include "core/board/board.h"

namespace rune {

struct ContextSpec {
  static constexpr int kDim = 12;
  static const char* names[12];
};

void computeContext(const Board& board, float* ctx);

}
