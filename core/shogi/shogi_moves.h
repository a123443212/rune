#pragma once

#include <cstdint>
#include <vector>

#include "core/shogi/shogi_board.h"

namespace rune {
namespace shogi {

struct ShogiMove {
  bool drop = false;
  int fr = -1;
  int to = -1;
  bool promo = false;
  uint8_t piece = 0;
};

uint8_t shogiUnpromote(uint8_t kind);
void shogiPseudoMoves(const ShogiBoard& board, std::vector<ShogiMove>& out);

}
}
