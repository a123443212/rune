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
    case 8: return 4560;
    default: return 0;
  }
}
int GroupedFeatureSet::pairIndex(int ida, int idb) {
  if (ida == idb) return -1;
  int lo = ida < idb ? ida : idb;
  int hi = ida < idb ? idb : ida;
  int idx = hi * (hi - 1) / 2 + lo;
  if (idx < 0 || idx >= kPairVocab) return -1;
  return idx;
}

const char* GroupedFeatureSet::version() { return "grouped_hkav2_fullthreats_v02"; }

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
  out.reserve(512);
  Color us = board.sideToMove();
  int kus = -1;
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = board.at(sq);
    if (!p.empty() && p.type == PieceType::King && p.color == us) {
      kus = sq;
      break;
    }
  }
  bool fold = kus >= 0 && fileOf(kus) < 4;
  auto rel = [fold](int sq) {
    if (!fold) return sq;
    return rankOf(sq) * 8 + (7 - fileOf(sq));
  };
  auto relColor = [us](Color c) { return (c == us) ? 0 : 1; };
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = board.at(sq);
    if (p.empty()) continue;
    int ci = relColor(p.color);
    int ti = typeIndex(p.type);
    int s = rel(sq);
    if (p.type == PieceType::Pawn) {
      out.push_back(ActiveFeature{0, static_cast<uint16_t>(ci * 64 + s)});
      int bucket = rankOf(s) / 2;
      out.push_back(ActiveFeature{0, static_cast<uint16_t>(128 + fileOf(s) * 4 + bucket)});
    }
    if (p.type == PieceType::King) {
      out.push_back(ActiveFeature{1, static_cast<uint16_t>(ci * 64 + s)});
      for (int df = -1; df <= 1; ++df) {
        for (int dr = -1; dr <= 1; ++dr) {
          if (df == 0 && dr == 0) continue;
          int f = fileOf(sq) + df;
          int r = rankOf(sq) + dr;
          if (!onBoard(f, r)) continue;
          int nsq = rel(makeSq(f, r));
          out.push_back(ActiveFeature{1, static_cast<uint16_t>(128 + ci * 32 + nsq / 2)});
        }
      }
    }
    if (p.type == PieceType::Knight) {
      out.push_back(ActiveFeature{2, static_cast<uint16_t>(ci * 64 + s)});
    }
    if (p.type == PieceType::Bishop) {
      out.push_back(ActiveFeature{2, static_cast<uint16_t>(128 + ci * 64 + s)});
    }
    if (p.type == PieceType::Rook) {
      out.push_back(ActiveFeature{3, static_cast<uint16_t>(ci * 64 + s)});
    }
    if (p.type == PieceType::Queen) {
      out.push_back(ActiveFeature{4, static_cast<uint16_t>(ci * 64 + s)});
    }
    if (ti >= 0) {
      out.push_back(ActiveFeature{6, static_cast<uint16_t>(ti * 64 + s)});
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
      out.push_back(ActiveFeature{5, static_cast<uint16_t>(vt * 64 + rel(vsq))});
      int coarse = 384 + fileOf(rel(asq)) * 8 + rankOf(asq);
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
  int epsq = board.epSquare();
  if (epsq >= 0) {
    out.push_back(ActiveFeature{7, static_cast<uint16_t>(37 + fileOf(rel(epsq)))});
  }
  int half = static_cast<int>(board.halfmoveClock());
  int hbuck = half / 20;
  if (hbuck > 4) hbuck = 4;
  out.push_back(ActiveFeature{7, static_cast<uint16_t>(45 + hbuck)});
  int pairIds[32];
  int pairCount = 0;
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = board.at(sq);
    if (p.empty() || p.type != PieceType::Pawn) continue;
    if (sq < 8 || sq > 55) continue;
    if (pairCount < 32) {
      pairIds[pairCount++] = relColor(p.color) * 48 + (rel(sq) - 8);
    }
  }
  for (int i = 0; i < pairCount; ++i) {
    for (int j = i + 1; j < pairCount; ++j) {
      int idx = pairIndex(pairIds[i], pairIds[j]);
      if (idx >= 0) out.push_back(ActiveFeature{8, static_cast<uint16_t>(idx)});
    }
  }
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
