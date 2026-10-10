#pragma once

#include <string>
#include <vector>

#include "core/xiangqi/xiangqi_board.h"

namespace rune {
namespace xiangqi {

std::string xiangqiEncodeMove(int fr, int to);
bool xiangqiDecodeMove(const std::string& mv, int& fr, int& to);
bool xiangqiKingInCheck(const XiangqiBoard& board, uint8_t color);
bool xiangqiLegalMoves(const std::string& state, std::vector<std::string>& out);
bool xiangqiApplyMove(const std::string& state, const std::string& mv, std::string& out);
std::string xiangqiGameResult(const std::string& state);

}
}
