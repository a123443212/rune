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
