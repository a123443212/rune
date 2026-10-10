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

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rune {

enum class Color : uint8_t { White = 0, Black = 1 };

enum class PieceType : uint8_t {
  None = 0,
  Pawn = 1,
  Knight = 2,
  Bishop = 3,
  Rook = 4,
  Queen = 5,
  King = 6
};

struct Piece {
  PieceType type = PieceType::None;
  Color color = Color::White;
  bool empty() const { return type == PieceType::None; }
};

inline Color opposite(Color c) { return c == Color::White ? Color::Black : Color::White; }

inline int fileOf(int sq) { return sq & 7; }
inline int rankOf(int sq) { return sq >> 3; }
inline int makeSq(int file, int rank) { return rank * 8 + file; }
inline bool onBoard(int file, int rank) { return file >= 0 && file < 8 && rank >= 0 && rank < 8; }

struct Move {
  uint8_t from = 0;
  uint8_t to = 0;
  PieceType promotion = PieceType::None;
  bool operator==(const Move& o) const {
    return from == o.from && to == o.to && promotion == o.promotion;
  }
};

struct BoardSnapshot {
  Piece captured;
  uint8_t castling = 0;
  int epSquarePrev = -1;
  uint16_t halfmove = 0;
  uint16_t fullmove = 1;
  Move move;
  bool isCastle = false;
  bool isEp = false;
};

class Board {
 public:
  Board();
  explicit Board(const std::string& fen);

  bool setFen(const std::string& fen);
  std::string toFen() const;

  Piece at(int sq) const { return squares_[sq]; }
  Color sideToMove() const { return side_; }
  uint8_t castling() const { return castling_; }
  int epSquare() const { return epSquare_; }
  uint16_t halfmoveClock() const { return halfmove_; }

  int kingSquare(Color c) const;
  bool isAttacked(int sq, Color by) const;
  bool inCheck(Color c) const;

  void generatePseudoLegalMoves(std::vector<Move>& out) const;
  void generateLegalMoves(std::vector<Move>& out);

  bool makeMove(const Move& m);
  void unmakeMove();

  bool isLegalPosition() const;
  uint64_t hashKey() const;
  int gamePhase() const;
  int pieceCount() const;
  size_t historySize() const { return history_.size(); }

  static constexpr uint8_t kCastleWK = 1;
  static constexpr uint8_t kCastleWQ = 2;
  static constexpr uint8_t kCastleBK = 4;
  static constexpr uint8_t kCastleBQ = 8;

 private:
  std::array<Piece, 64> squares_;
  Color side_ = Color::White;
  uint8_t castling_ = 0;
  int epSquare_ = -1;
  uint16_t halfmove_ = 0;
  uint16_t fullmove_ = 1;
  std::vector<BoardSnapshot> history_;

  void addPawnMoves(int sq, std::vector<Move>& out) const;
  void addKnightMoves(int sq, std::vector<Move>& out) const;
  void addKingMoves(int sq, std::vector<Move>& out) const;
  void addSlidingMoves(int sq, std::vector<Move>& out, bool diag, bool straight) const;
  void applyMove(const Move& m, Piece& captured);
  void moveRookForCastle(int kingTo);
  void dropCastlingRight(const Piece& moving, const Move& m, const Piece& captured);
};

}
