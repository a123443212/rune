#include "core/architectures/relational/context.h"

namespace rune {

const char* ContextSpec::names[12] = {"stm",      "phase",  "pawns", "minors",
                                      "rooks",    "queens", "shield", "total",
                                      "castle",   "ep",     "check",  "halfmove"};

void computeContext(const Board& board, float* ctx) {
  int stm = (board.sideToMove() == Color::White) ? 0 : 1;
  Color us = board.sideToMove();
  int pawns = 0;
  int minors = 0;
  int rooks = 0;
  int queens = 0;
  int total = 0;
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = board.at(sq);
    if (p.empty()) continue;
    total += 1;
    if (p.type == PieceType::Pawn) pawns += 1;
    if (p.type == PieceType::Knight || p.type == PieceType::Bishop) minors += 1;
    if (p.type == PieceType::Rook) rooks += 1;
    if (p.type == PieceType::Queen) queens += 1;
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
  ctx[2] = clip01(static_cast<float>(pawns) / 16.0f);
  ctx[3] = clip01(static_cast<float>(minors) / 8.0f);
  ctx[4] = clip01(static_cast<float>(rooks) / 4.0f);
  ctx[5] = clip01(static_cast<float>(queens) / 2.0f);
  ctx[6] = clip01(static_cast<float>(shield) / 8.0f);
  ctx[7] = clip01(static_cast<float>(total) / 32.0f);
  ctx[8] = clip01(static_cast<float>(board.castling()) / 15.0f);
  ctx[9] = (board.epSquare() >= 0) ? 1.0f : 0.0f;
  int inCheck = 0;
  if (king >= 0 && board.isAttacked(king, opposite(us))) {
    inCheck = 1;
  }
  ctx[10] = static_cast<float>(inCheck);
  int half = static_cast<int>(board.halfmoveClock());
  if (half < 0) half = 0;
  ctx[11] = clip01(static_cast<float>(half) / 100.0f);
}

}
