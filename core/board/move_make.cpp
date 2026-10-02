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
    captured = squares_[capSq];
    squares_[capSq] = Piece{};
  }
  bool isCastle = (moving.type == PieceType::King &&
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
  if (kingTo == makeSq(6, 0)) {
    squares_[makeSq(5, 0)] = squares_[makeSq(7, 0)];
    squares_[makeSq(7, 0)] = Piece{};
  } else if (kingTo == makeSq(2, 0)) {
    squares_[makeSq(3, 0)] = squares_[makeSq(0, 0)];
    squares_[makeSq(0, 0)] = Piece{};
  } else if (kingTo == makeSq(6, 7)) {
    squares_[makeSq(5, 7)] = squares_[makeSq(7, 7)];
    squares_[makeSq(7, 7)] = Piece{};
  } else if (kingTo == makeSq(2, 7)) {
    squares_[makeSq(3, 7)] = squares_[makeSq(0, 7)];
    squares_[makeSq(0, 7)] = Piece{};
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
  }
  BoardSnapshot snap;
  snap.move = m;
  snap.captured = squares_[m.to];
  snap.castling = castling_;
  snap.epSquarePrev = epSquare_;
  snap.halfmove = halfmove_;
  snap.fullmove = fullmove_;
  snap.isCastle = isCastle;
  snap.isEp = (moving.type == PieceType::Pawn && m.to == epSquare_ && squares_[m.to].empty() &&
               fileOf(m.from) != fileOf(m.to));
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
