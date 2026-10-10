#pragma once

#include <string>
#include <vector>

#include "core/shogi/shogi_board.h"
#include "core/shogi/shogi_moves.h"

namespace rune {
namespace shogi {

std::string shogiEncodeMove(ShogiMove mv);
bool shogiDecodeMove(const std::string& s, ShogiMove& mv);
bool shogiKingInCheck(const ShogiBoard& board, uint8_t color);
bool shogiLegalMoves(const std::string& sfen, std::vector<std::string>& out);
bool shogiApplyMove(const std::string& sfen, const std::string& mv, std::string& out);
std::string shogiGameResult(const std::string& sfen);

}
}
