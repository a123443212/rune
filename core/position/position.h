#pragma once

#include <cstdint>
#include <string>

#include "core/board/board.h"

namespace rune {

struct PositionMeta {
  std::string gameId;
  int ply = 0;
  int phase = 1;
  std::string source;
};

class Position {
 public:
  Position();
  explicit Position(const std::string& fen);

  bool set(const std::string& fen, const PositionMeta& meta = PositionMeta{});
  const Board& board() const { return board_; }
  Board& board() { return board_; }
  const PositionMeta& meta() const { return meta_; }

  bool makeMove(const Move& m);
  void unmakeMove();
  void generateLegalMoves(std::vector<Move>& out);

  std::string normalizedKey() const;

 private:
  Board board_;
  PositionMeta meta_;
};

}
