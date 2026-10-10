#pragma once

#include <vector>

#include "core/xiangqi/xiangqi_board.h"

namespace rune {
namespace xiangqi {

void xiangqiPseudoDests(const XiangqiBoard& board, int sq, std::vector<int>& out);

}
}
