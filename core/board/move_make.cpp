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

#include "core/board/board.h"

namespace rune {

void Board::applyMove(const Move& m, Piece& captured) {
  Piece moving = squares_[m.from];
  captured = squares_[m.to];
  bool isEp = (moving.type == PieceType::Pawn && m.to == epSquare_ && captured.empty());
  squares_[m.from] = Piece{};
  Piece placed = moving;
  if (m.promotion != PieceType::None) placed.type = m.promotion;
  squares_[m.to] = placed;
  if (isEp) {
    int capSq = makeSq(fileOf(m.to), rankOf(m.from));
    Piece cap = squares_[capSq];
    if (!cap.empty() && cap.type == PieceType::Pawn && cap.color != moving.color) {
      captured = cap;
      squares_[capSq] = Piece{};
    }
  }
  bool isCastle = (moving.type == PieceType::King && rankOf(m.from) == rankOf(m.to) &&
                   (m.to == m.from + 2 || (m.to + 2 == m.from)));
  if (isCastle) moveRookForCastle(m.to);
  dropCastlingRight(moving, m, captured);
  epSquare_ = -1;
  if (moving.type == PieceType::Pawn && (m.to == m.from + 16 || m.from == m.to + 16)) {
    epSquare_ = (m.from + m.to) / 2;
  }
  if (moving.color == Color::Black) fullmove_ += 1;
  if (moving.type == PieceType::Pawn || !captured.empty()) halfmove_ = 0;
  else halfmove_ += 1;
  side_ = opposite(side_);
}

void Board::moveRookForCastle(int kingTo) {
  auto moveRook = [&](int from, int to) {
    Piece r = squares_[from];
    if (r.empty() || r.type != PieceType::Rook) return;
    squares_[to] = r;
    squares_[from] = Piece{};
  };
  if (kingTo == makeSq(6, 0)) {
    moveRook(makeSq(7, 0), makeSq(5, 0));
  } else if (kingTo == makeSq(2, 0)) {
    moveRook(makeSq(0, 0), makeSq(3, 0));
  } else if (kingTo == makeSq(6, 7)) {
    moveRook(makeSq(7, 7), makeSq(5, 7));
  } else if (kingTo == makeSq(2, 7)) {
    moveRook(makeSq(0, 7), makeSq(3, 7));
  }
}

void Board::dropCastlingRight(const Piece& moving, const Move& m, const Piece& captured) {
  if (moving.type == PieceType::King) {
    if (moving.color == Color::White) castling_ &= static_cast<uint8_t>(~(kCastleWK | kCastleWQ));
    else castling_ &= static_cast<uint8_t>(~(kCastleBK | kCastleBQ));
  }
  auto dropOnSquare = [&](int sq, uint8_t flag) {
    if (m.from == sq || (captured.type == PieceType::Rook && m.to == sq)) {
      castling_ &= static_cast<uint8_t>(~flag);
    }
  };
  if (moving.type == PieceType::Rook || captured.type == PieceType::Rook) {
    dropOnSquare(makeSq(0, 0), kCastleWQ);
    dropOnSquare(makeSq(7, 0), kCastleWK);
    dropOnSquare(makeSq(0, 7), kCastleBQ);
    dropOnSquare(makeSq(7, 7), kCastleBK);
  }
}

bool Board::makeMove(const Move& m) {
  Piece moving = squares_[m.from];
  if (moving.empty() || moving.color != side_) return false;
  bool isCastle = (moving.type == PieceType::King && rankOf(m.from) == rankOf(m.to) &&
                   (m.to == m.from + 2 || m.to + 2 == m.from));
  if (isCastle) {
    int pass = (m.to > m.from) ? m.from + 1 : m.from - 1;
    Color enemy = opposite(side_);
    if (inCheck(side_) || isAttacked(pass, enemy)) return false;
    int rookFrom = -1;
    if (m.to == makeSq(6, 0)) rookFrom = makeSq(7, 0);
    else if (m.to == makeSq(2, 0)) rookFrom = makeSq(0, 0);
    else if (m.to == makeSq(6, 7)) rookFrom = makeSq(7, 7);
    else if (m.to == makeSq(2, 7)) rookFrom = makeSq(0, 7);
    else return false;
    Piece rook = squares_[rookFrom];
    if (rook.empty() || rook.type != PieceType::Rook || rook.color != side_) return false;
  }
  bool epAttempt = (moving.type == PieceType::Pawn && m.to == epSquare_ && squares_[m.to].empty() &&
               fileOf(m.from) != fileOf(m.to));
  if (epAttempt) {
    Piece cap = squares_[makeSq(fileOf(m.to), rankOf(m.from))];
    if (cap.empty() || cap.type != PieceType::Pawn || cap.color == side_) return false;
  }
  BoardSnapshot snap;
  snap.move = m;
  snap.captured = squares_[m.to];
  snap.castling = castling_;
  snap.epSquarePrev = epSquare_;
  snap.isEp = epAttempt;
  if (snap.isEp) {
    snap.captured = squares_[makeSq(fileOf(m.to), rankOf(m.from))];
  }
  snap.halfmove = halfmove_;
  snap.fullmove = fullmove_;
  snap.isCastle = isCastle;
  Color us = side_;
  std::array<Piece, 64> saved = squares_;
  uint8_t savedCastling = castling_;
  int savedEp = epSquare_;
  uint16_t savedHalf = halfmove_;
  uint16_t savedFull = fullmove_;
  Piece ignored;
  applyMove(m, ignored);
  if (inCheck(us)) {
    squares_ = saved;
    castling_ = savedCastling;
    epSquare_ = savedEp;
    halfmove_ = savedHalf;
    fullmove_ = savedFull;
    side_ = us;
    return false;
  }
  history_.push_back(snap);
  return true;
}

void Board::unmakeMove() {
  if (history_.empty()) return;
  BoardSnapshot snap = history_.back();
  history_.pop_back();
  side_ = opposite(side_);
  Move m = snap.move;
  Piece moving = squares_[m.to];
  if (m.promotion != PieceType::None) moving.type = PieceType::Pawn;
  squares_[m.from] = moving;
  squares_[m.to] = Piece{};
  if (snap.isCastle) {
    if (m.to == makeSq(6, 0)) {
      squares_[makeSq(7, 0)] = squares_[makeSq(5, 0)];
      squares_[makeSq(5, 0)] = Piece{};
    } else if (m.to == makeSq(2, 0)) {
      squares_[makeSq(0, 0)] = squares_[makeSq(3, 0)];
      squares_[makeSq(3, 0)] = Piece{};
    } else if (m.to == makeSq(6, 7)) {
      squares_[makeSq(7, 7)] = squares_[makeSq(5, 7)];
      squares_[makeSq(5, 7)] = Piece{};
    } else if (m.to == makeSq(2, 7)) {
      squares_[makeSq(0, 7)] = squares_[makeSq(3, 7)];
      squares_[makeSq(3, 7)] = Piece{};
    }
  } else if (snap.isEp) {
    squares_[m.to] = Piece{};
    squares_[makeSq(fileOf(m.to), rankOf(m.from))] = snap.captured;
  } else {
    squares_[m.to] = snap.captured;
  }
  castling_ = snap.castling;
  halfmove_ = snap.halfmove;
  fullmove_ = snap.fullmove;
  epSquare_ = snap.epSquarePrev;
}

}
