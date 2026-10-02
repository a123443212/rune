#include "core/features/feature_set.h"

#include <algorithm>

namespace rune {

namespace {

int typeIndex(PieceType t) {
  switch (t) {
    case PieceType::Pawn: return 0;
    case PieceType::Knight: return 1;
    case PieceType::Bishop: return 2;
    case PieceType::Rook: return 3;
    case PieceType::Queen: return 4;
    case PieceType::King: return 5;
    default: return -1;
  }
}

bool slidingAttacks(const Board& b, int from, int target) {
  Piece p = b.at(from);
  int df = fileOf(target) - fileOf(from);
  int dr = rankOf(target) - rankOf(from);
  int adf = df < 0 ? -df : df;
  int adr = dr < 0 ? -dr : dr;
  bool diag = false;
  bool straight = false;
  if (p.type == PieceType::Bishop) diag = true;
  if (p.type == PieceType::Rook) straight = true;
  if (p.type == PieceType::Queen) {
    diag = true;
    straight = true;
  }
  if (!diag && !straight) return false;
  bool isDiag = (adf == adr);
  bool isStraight = (df == 0 || dr == 0);
  if (isDiag && !diag) return false;
  if (isStraight && !straight) return false;
  if (!isDiag && !isStraight) return false;
  int stepF = (df == 0) ? 0 : (df > 0 ? 1 : -1);
  int stepR = (dr == 0) ? 0 : (dr > 0 ? 1 : -1);
  int f = fileOf(from) + stepF;
  int r = rankOf(from) + stepR;
  while (f != fileOf(target) || r != rankOf(target)) {
    if (!b.at(makeSq(f, r)).empty()) return false;
    f += stepF;
    r += stepR;
  }
  return true;
}

bool pieceAttacks(const Board& b, int from, int target) {
  Piece p = b.at(from);
  if (p.empty()) return false;
  int df = fileOf(target) - fileOf(from);
  int dr = rankOf(target) - rankOf(from);
  int adf = df < 0 ? -df : df;
  int adr = dr < 0 ? -dr : dr;
  switch (p.type) {
    case PieceType::Pawn: {
      int d = (p.color == Color::White) ? 1 : -1;
      return adr == 1 && adf == 1 && dr == d;
    }
    case PieceType::Knight:
      return (adf == 1 && adr == 2) || (adf == 2 && adr == 1);
    case PieceType::King:
      return adf <= 1 && adr <= 1;
    case PieceType::Bishop:
    case PieceType::Rook:
    case PieceType::Queen:
      return slidingAttacks(b, from, target);
    default:
      return false;
  }
}

}  // namespace

int GroupedFeatureSet::vocabSize(int group) {
  switch (group) {
    case 0: return 256;
    case 1: return 256;
    case 2: return 256;
    case 3: return 128;
    case 4: return 128;
    case 5: return 512;
    case 6: return 512;
    case 7: return 64;
    default: return 0;
  }
}

const char* GroupedFeatureSet::version() { return "grouped_hkav2_fullthreats_v01"; }

const char* GroupedFeatureSet::groupName(int group) {
  switch (group) {
    case 0: return "pawn_structure";
    case 1: return "king_zone";
    case 2: return "minor_pieces";
    case 3: return "rooks";
    case 4: return "queens";
    case 5: return "threats";
    case 6: return "mobility";
    case 7: return "global";
    default: return "unknown";
  }
}

void GroupedFeatureSet::extract(const Board& board, std::vector<ActiveFeature>& out) {
  out.clear();
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = board.at(sq);
    if (p.empty()) continue;
    int ci = (p.color == Color::White) ? 0 : 1;
    int ti = typeIndex(p.type);
    if (p.type == PieceType::Pawn) {
      out.push_back(ActiveFeature{0, static_cast<uint16_t>(ci * 64 + sq)});
      int bucket = rankOf(sq) / 2;
      out.push_back(ActiveFeature{0, static_cast<uint16_t>(128 + fileOf(sq) * 4 + bucket)});
    }
    if (p.type == PieceType::King) {
      out.push_back(ActiveFeature{1, static_cast<uint16_t>(ci * 64 + sq)});
      for (int df = -1; df <= 1; ++df) {
        for (int dr = -1; dr <= 1; ++dr) {
          if (df == 0 && dr == 0) continue;
          int f = fileOf(sq) + df;
          int r = rankOf(sq) + dr;
          if (!onBoard(f, r)) continue;
          int nsq = makeSq(f, r);
          out.push_back(ActiveFeature{1, static_cast<uint16_t>(128 + ci * 32 + nsq / 2)});
        }
      }
    }
    if (p.type == PieceType::Knight) {
      out.push_back(ActiveFeature{2, static_cast<uint16_t>(ci * 64 + sq)});
    }
    if (p.type == PieceType::Bishop) {
      out.push_back(ActiveFeature{2, static_cast<uint16_t>(128 + ci * 64 + sq)});
    }
    if (p.type == PieceType::Rook) {
      out.push_back(ActiveFeature{3, static_cast<uint16_t>(ci * 64 + sq)});
    }
    if (p.type == PieceType::Queen) {
      out.push_back(ActiveFeature{4, static_cast<uint16_t>(ci * 64 + sq)});
    }
    if (ti >= 0) {
      out.push_back(ActiveFeature{6, static_cast<uint16_t>(ti * 64 + sq)});
    }
  }
  for (int vsq = 0; vsq < 64; ++vsq) {
    Piece victim = board.at(vsq);
    if (victim.empty()) continue;
    for (int asq = 0; asq < 64; ++asq) {
      Piece attacker = board.at(asq);
      if (attacker.empty() || attacker.color == victim.color) continue;
      if (!pieceAttacks(board, asq, vsq)) continue;
      int vt = typeIndex(victim.type);
      out.push_back(ActiveFeature{5, static_cast<uint16_t>(vt * 64 + vsq)});
      int coarse = 384 + fileOf(asq) * 8 + rankOf(asq);
      if (coarse < 512) out.push_back(ActiveFeature{5, static_cast<uint16_t>(coarse)});
    }
  }
  std::vector<Move> pseudo;
  board.generatePseudoLegalMoves(pseudo);
  int mob = static_cast<int>(pseudo.size());
  if (mob > 31) mob = 31;
  int stm = (board.sideToMove() == Color::White) ? 0 : 1;
  out.push_back(ActiveFeature{6, static_cast<uint16_t>(384 + stm * 32 + mob)});
  out.push_back(ActiveFeature{7, static_cast<uint16_t>(stm)});
  out.push_back(ActiveFeature{7, static_cast<uint16_t>(2 + board.castling())});
  int total = board.pieceCount();
  int bucket = (total - 2) / 2;
  if (bucket < 0) bucket = 0;
  if (bucket > 15) bucket = 15;
  out.push_back(ActiveFeature{7, static_cast<uint16_t>(18 + bucket)});
  out.push_back(ActiveFeature{7, static_cast<uint16_t>(34 + board.gamePhase())});
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
}

void GroupedFeatureSet::diffFeatures(const std::vector<ActiveFeature>& before,
                                     const std::vector<ActiveFeature>& after,
                                     std::vector<ActiveFeature>& added,
                                     std::vector<ActiveFeature>& removed) {
  added.clear();
  removed.clear();
  size_t i = 0;
  size_t j = 0;
  while (i < before.size() && j < after.size()) {
    if (before[i] == after[j]) {
      ++i;
      ++j;
    } else if (after[j] < before[i]) {
      added.push_back(after[j]);
      ++j;
    } else {
      removed.push_back(before[i]);
      ++i;
    }
  }
  while (j < after.size()) added.push_back(after[j++]);
  while (i < before.size()) removed.push_back(before[i++]);
}

}
