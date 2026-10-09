#pragma once

#include <vector>

#include "core/go/go_board.h"

namespace rune {
namespace go {

void extractPlanes(const GoBoard& board, std::vector<float>& out);
void extractPlanesInto(const GoBoard& board, float* out);

}
}
