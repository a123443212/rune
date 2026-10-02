#pragma once

#include <array>

#include "core/board/board.h"

namespace rune {

inline int dirRank(Color c) { return c == Color::White ? 1 : -1; }

bool attacksSquare(const std::array<Piece, 64>& squares, int from, int target);

}
