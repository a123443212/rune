/*
RUNE — Relational Unified Neural Evaluator
Copyright (C) 2026 a123443212

SPDX-License-Identifier: MIT OR Apache-2.0

This project is dual-licensed under the MIT License and the
Apache License, Version 2.0. You may choose either license
when using, copying, modifying, or distributing this software.

MIT License: https://opensource.org/license/mit
Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, this
software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
*/

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
