#include "core/board/board.h"

#include "core/board/board_int.h"

namespace rune {

void Board::addPawnMoves(int sq, std::vector<Move>& out) const {
  Piece p = squares_[sq];
  int d = dirRank(p.color);
  int f = fileOf(sq);
  int r = rankOf(sq);
  int startRank = (p.color == Color::White) ? 1 : 6;
  int promoRank = (p.color == Color::White) ? 7 : 0;
  auto pushPromo = [&](int from, int to) {
    if (rankOf(to) == promoRank) {
      for (PieceType t : {PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight}) {
        out.push_back(Move{static_cast<uint8_t>(from), static_cast<uint8_t>(to), t});
      }
    } else {
      out.push_back(Move{static_cast<uint8_t>(from), static_cast<uint8_t>(to), PieceType::None});
    }
  };
  if (onBoard(f, r + d) && squares_[makeSq(f, r + d)].empty()) {
    pushPromo(sq, makeSq(f, r + d));
    if (r == startRank && squares_[makeSq(f, r + 2 * d)].empty()) {
      out.push_back(Move{static_cast<uint8_t>(sq), static_cast<uint8_t>(makeSq(f, r + 2 * d)), PieceType::None});
    }
  }
  for (int df : {-1, 1}) {
    if (!onBoard(f + df, r + d)) continue;
    int to = makeSq(f + df, r + d);
    if (!squares_[to].empty() && squares_[to].color != p.color) pushPromo(sq, to);
    if (to == epSquare_) out.push_back(Move{static_cast<uint8_t>(sq), static_cast<uint8_t>(to), PieceType::None});
  }
}

void Board::addSlidingMoves(int sq, std::vector<Move>& out, bool diag, bool straight) const {
  Piece p = squares_[sq];
  for (int df = -1; df <= 1; ++df) {
    for (int dr = -1; dr <= 1; ++dr) {
      if (df == 0 && dr == 0) continue;
      bool isDiag = (df != 0 && dr != 0);
      if (isDiag && !diag) continue;
      if (!isDiag && !straight) continue;
      int f = fileOf(sq) + df;
      int r = rankOf(sq) + dr;
      while (onBoard(f, r)) {
        int to = makeSq(f, r);
        if (squares_[to].empty()) {
          out.push_back(Move{static_cast<uint8_t>(sq), static_cast<uint8_t>(to), PieceType::None});
        } else {
          if (squares_[to].color != p.color) {
            out.push_back(Move{static_cast<uint8_t>(sq), static_cast<uint8_t>(to), PieceType::None});
          }
          break;
        }
        f += df;
        r += dr;
      }
    }
  }
}

void Board::addKnightMoves(int sq, std::vector<Move>& out) const {
  static const int kSteps[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
  Piece p = squares_[sq];
  for (auto d : kSteps) {
    if (!onBoard(fileOf(sq) + d[0], rankOf(sq) + d[1])) continue;
    int to = makeSq(fileOf(sq) + d[0], rankOf(sq) + d[1]);
    if (squares_[to].empty() || squares_[to].color != p.color) {
      out.push_back(Move{static_cast<uint8_t>(sq), static_cast<uint8_t>(to), PieceType::None});
    }
  }
}

void Board::addKingMoves(int sq, std::vector<Move>& out) const {
  Piece p = squares_[sq];
  for (int df = -1; df <= 1; ++df) {
    for (int dr = -1; dr <= 1; ++dr) {
      if (df == 0 && dr == 0) continue;
      if (!onBoard(fileOf(sq) + df, rankOf(sq) + dr)) continue;
      int to = makeSq(fileOf(sq) + df, rankOf(sq) + dr);
      if (squares_[to].empty() || squares_[to].color != p.color) {
        out.push_back(Move{static_cast<uint8_t>(sq), static_cast<uint8_t>(to), PieceType::None});
      }
    }
  }
  if (p.color == Color::White && sq == makeSq(4, 0)) {
    if ((castling_ & kCastleWK) && squares_[makeSq(5, 0)].empty() && squares_[makeSq(6, 0)].empty()) {
      out.push_back(Move{4, 6, PieceType::None});
    }
    if ((castling_ & kCastleWQ) && squares_[makeSq(3, 0)].empty() && squares_[makeSq(2, 0)].empty() &&
        squares_[makeSq(1, 0)].empty()) {
      out.push_back(Move{4, 2, PieceType::None});
    }
  }
  if (p.color == Color::Black && sq == makeSq(4, 7)) {
    if ((castling_ & kCastleBK) && squares_[makeSq(5, 7)].empty() && squares_[makeSq(6, 7)].empty()) {
      out.push_back(Move{60, 62, PieceType::None});
    }
    if ((castling_ & kCastleBQ) && squares_[makeSq(3, 7)].empty() && squares_[makeSq(2, 7)].empty() &&
        squares_[makeSq(1, 7)].empty()) {
      out.push_back(Move{60, 58, PieceType::None});
    }
  }
}

void Board::generatePseudoLegalMoves(std::vector<Move>& out) const {
  out.clear();
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = squares_[sq];
    if (p.empty() || p.color != side_) continue;
    switch (p.type) {
      case PieceType::Pawn: addPawnMoves(sq, out); break;
      case PieceType::Knight: addKnightMoves(sq, out); break;
      case PieceType::Bishop: addSlidingMoves(sq, out, true, false); break;
      case PieceType::Rook: addSlidingMoves(sq, out, false, true); break;
      case PieceType::Queen: addSlidingMoves(sq, out, true, true); break;
      case PieceType::King: addKingMoves(sq, out); break;
      default: break;
    }
  }
}

void Board::generateLegalMoves(std::vector<Move>& out) {
  std::vector<Move> pseudo;
  generatePseudoLegalMoves(pseudo);
  out.clear();
  for (const Move& m : pseudo) {
    if (makeMove(m)) {
      unmakeMove();
      out.push_back(m);
    }
  }
}

}
