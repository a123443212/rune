#include "core/position/position.h"

namespace rune {

Position::Position() : board_() {
  meta_.phase = board_.gamePhase();
}

Position::Position(const std::string& fen) : board_(fen) {
  meta_.phase = board_.gamePhase();
}

bool Position::set(const std::string& fen, const PositionMeta& meta) {
  if (!board_.setFen(fen)) return false;
  meta_ = meta;
  meta_.phase = board_.gamePhase();
  return true;
}

bool Position::makeMove(const Move& m) {
  if (!board_.makeMove(m)) return false;
  meta_.ply += 1;
  meta_.phase = board_.gamePhase();
  return true;
}

void Position::unmakeMove() {
  if (board_.historySize() == 0) {
    return;
  }
  board_.unmakeMove();
  meta_.ply -= 1;
  meta_.phase = board_.gamePhase();
}

void Position::generateLegalMoves(std::vector<Move>& out) {
  board_.generateLegalMoves(out);
}

std::string Position::normalizedKey() const {
  std::string fen = board_.toFen();
  auto pos = fen.find_last_of(' ');
  auto prev = fen.find_last_of(' ', pos - 1);
  return fen.substr(0, prev);
}

}
