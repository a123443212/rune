#include "core/architectures/relational/context.h"

namespace rune {

const char* ContextSpec::names[17] = {"stm",      "phase",    "us_pawns", "them_pawns",
                                        "us_minors", "them_minors", "us_rooks", "them_rooks",
                                        "us_queens", "them_queens", "us_total", "them_total",
                                        "shield",   "castle",      "ep",       "check",
                                        "halfmove"};

void computeContext(const Board& board, float* ctx) {
  int stm = (board.sideToMove() == Color::White) ? 0 : 1;
  Color us = board.sideToMove();
  Color them = opposite(us);
  int usPawns = 0;
  int themPawns = 0;
  int usMinors = 0;
  int themMinors = 0;
  int usRooks = 0;
  int themRooks = 0;
  int usQueens = 0;
  int themQueens = 0;
  int usTotal = 0;
  int themTotal = 0;
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = board.at(sq);
    if (p.empty()) continue;
    if (p.color == us) {
      usTotal += 1;
      if (p.type == PieceType::Pawn) usPawns += 1;
      if (p.type == PieceType::Knight || p.type == PieceType::Bishop) usMinors += 1;
      if (p.type == PieceType::Rook) usRooks += 1;
      if (p.type == PieceType::Queen) usQueens += 1;
    } else {
      themTotal += 1;
      if (p.type == PieceType::Pawn) themPawns += 1;
      if (p.type == PieceType::Knight || p.type == PieceType::Bishop) themMinors += 1;
      if (p.type == PieceType::Rook) themRooks += 1;
      if (p.type == PieceType::Queen) themQueens += 1;
    }
  }
  int shield = 0;
  int king = board.kingSquare(us);
  if (king >= 0) {
    for (int df = -1; df <= 1; ++df) {
      for (int dr = -1; dr <= 1; ++dr) {
        if (df == 0 && dr == 0) continue;
        int f = fileOf(king) + df;
        int r = rankOf(king) + dr;
        if (!onBoard(f, r)) continue;
        Piece p = board.at(makeSq(f, r));
        if (!p.empty() && p.color == us && p.type != PieceType::King) shield += 1;
      }
    }
  }
  auto clip01 = [](float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
  };
  ctx[0] = clip01(static_cast<float>(stm));
  ctx[1] = clip01(static_cast<float>(board.gamePhase()) / 2.0f);
  ctx[2] = clip01(static_cast<float>(usPawns) / 8.0f);
  ctx[3] = clip01(static_cast<float>(themPawns) / 8.0f);
  ctx[4] = clip01(static_cast<float>(usMinors) / 8.0f);
  ctx[5] = clip01(static_cast<float>(themMinors) / 8.0f);
  ctx[6] = clip01(static_cast<float>(usRooks) / 4.0f);
  ctx[7] = clip01(static_cast<float>(themRooks) / 4.0f);
  ctx[8] = clip01(static_cast<float>(usQueens) / 2.0f);
  ctx[9] = clip01(static_cast<float>(themQueens) / 2.0f);
  ctx[10] = clip01(static_cast<float>(usTotal) / 16.0f);
  ctx[11] = clip01(static_cast<float>(themTotal) / 16.0f);
  ctx[12] = clip01(static_cast<float>(shield) / 8.0f);
  ctx[13] = clip01(static_cast<float>(board.castling()) / 15.0f);
  ctx[14] = (board.epSquare() >= 0) ? 1.0f : 0.0f;
  int inCheck = 0;
  if (king >= 0 && board.isAttacked(king, them)) {
    inCheck = 1;
  }
  ctx[15] = static_cast<float>(inCheck);
  int half = static_cast<int>(board.halfmoveClock());
  if (half < 0) half = 0;
  ctx[16] = clip01(static_cast<float>(half) / 100.0f);
}

}
