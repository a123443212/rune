#include "core/board/board_int.h"

namespace rune {

bool attacksSquare(const std::array<Piece, 64>& s, int from, int target) {
  Piece p = s[from];
  if (p.empty()) return false;
  int df = fileOf(target) - fileOf(from);
  int dr = rankOf(target) - rankOf(from);
  int adf = df < 0 ? -df : df;
  int adr = dr < 0 ? -dr : dr;
  switch (p.type) {
    case PieceType::Pawn: {
      int d = dirRank(p.color);
      return adr == 1 && adf == 1 && dr == d;
    }
    case PieceType::Knight:
      return (adf == 1 && adr == 2) || (adf == 2 && adr == 1);
    case PieceType::Bishop:
      if (adf != adr) return false;
      break;
    case PieceType::Rook:
      if (df != 0 && dr != 0) return false;
      break;
    case PieceType::Queen:
      if (!((adf == adr) || df == 0 || dr == 0)) return false;
      break;
    case PieceType::King:
      return adf <= 1 && adr <= 1;
    default:
      return false;
  }
  int stepF = (df == 0) ? 0 : (df > 0 ? 1 : -1);
  int stepR = (dr == 0) ? 0 : (dr > 0 ? 1 : -1);
  int f = fileOf(from) + stepF;
  int r = rankOf(from) + stepR;
  while (f != fileOf(target) || r != rankOf(target)) {
    if (!s[makeSq(f, r)].empty()) return false;
    f += stepF;
    r += stepR;
  }
  return true;
}

}
