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

#include "core/shogi/shogi_features.h"

#include <algorithm>

namespace rune {
namespace shogi {

namespace {

const int kVocabs[9] = {648, 648, 972, 45, 45, 1134, 1134, 64, 1};
const int kHandMax[7] = {18, 4, 4, 4, 4, 2, 2};
const int kHandBase[7] = {0, 19, 24, 29, 34, 39, 42};

int minorIndex(uint8_t kind) {
  switch (kind) {
    case kPawn: return 0;
    case kLance: return 1;
    case kKnight: return 2;
    case kSilver: return 3;
    case kGold: return 4;
    case kBishop: return 5;
    case kRook: return 6;
    default: return 0;
  }
}

int phaseOf(uint32_t handTotal, uint32_t promoCount) {
  uint32_t n = handTotal + promoCount;
  if (n <= 3) return 0;
  if (n <= 9) return 1;
  return 2;
}

float clamp01(float v) {
  if (v < 0.0f) return 0.0f;
  if (v > 1.0f) return 1.0f;
  return v;
}

}  // namespace

int ShogiFeatureSet::vocabSize(int group) { return kVocabs[group]; }

const char* ShogiFeatureSet::version() { return "shogi_raw_v01"; }

void ShogiFeatureSet::extract(const ShogiBoard& board, std::vector<ActiveFeature>& out) {
  out.clear();
  uint8_t us = board.sideToMove();
  for (int sq = 0; sq < 81; ++sq) {
    ShogiPiece cell = board.at(sq);
    if (!cell.present) continue;
    uint16_t ci = (cell.color == us) ? 0 : 1;
    ActiveFeature f;
    if (cell.kind <= kSilver) {
      f.group = 0;
      f.index = static_cast<uint16_t>((ci * 4 + cell.kind) * 81 + sq);
      out.push_back(f);
    } else if (cell.kind <= kKing) {
      f.group = 1;
      f.index = static_cast<uint16_t>((ci * 4 + (cell.kind - kGold)) * 81 + sq);
      out.push_back(f);
    } else {
      f.group = 2;
      f.index = static_cast<uint16_t>((ci * 6 + (cell.kind - kProPawn)) * 81 + sq);
      out.push_back(f);
    }
    if (cell.kind <= kRook && cell.kind != kKing) {
      f.group = 6;
      f.index = static_cast<uint16_t>((ci * 7 + minorIndex(cell.kind)) * 81 + sq);
      out.push_back(f);
    }
  }
  for (int t = 0; t < 7; ++t) {
    int cu = board.hand(us, t);
    if (cu > kHandMax[t]) cu = kHandMax[t];
    int ct = board.hand(us ^ 1, t);
    if (ct > kHandMax[t]) ct = kHandMax[t];
    ActiveFeature fu;
    fu.group = 3;
    fu.index = static_cast<uint16_t>(kHandBase[t] + cu);
    out.push_back(fu);
    ActiveFeature ft;
    ft.group = 4;
    ft.index = static_cast<uint16_t>(kHandBase[t] + ct);
    out.push_back(ft);
  }
  int kus = -1;
  for (int sq = 0; sq < 81; ++sq) {
    ShogiPiece c = board.at(sq);
    if (c.present && c.kind == kKing && c.color == us) {
      kus = sq;
      break;
    }
  }
  bool check = false;
  if (kus >= 0) {
    for (int asq = 0; asq < 81; ++asq) {
      ShogiPiece a = board.at(asq);
      if (!a.present || a.color == us) continue;
      if (shogiAttacks(board, asq, kus)) {
        check = true;
        ActiveFeature f;
        f.group = 5;
        f.index = static_cast<uint16_t>(a.kind * 81 + asq);
        out.push_back(f);
      }
    }
  }
  uint32_t handTotal = board.handTotal();
  int ph = phaseOf(handTotal, board.promoCount());
  ActiveFeature f;
  f.group = 7;
  f.index = us;
  out.push_back(f);
  f.index = static_cast<uint16_t>(2 + (handTotal / 4 > 7 ? 7 : handTotal / 4));
  out.push_back(f);
  f.index = static_cast<uint16_t>(10 + ph);
  out.push_back(f);
  if (check) {
    f.index = 13;
    out.push_back(f);
  }
  uint32_t mv = board.moveNo() / 20;
  if (mv > 5) mv = 5;
  f.index = static_cast<uint16_t>(14 + mv);
  out.push_back(f);
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
}

void ShogiFeatureSet::context(const ShogiBoard& board, float* out) {
  uint8_t us = board.sideToMove();
  uint32_t usHand = 0, themHand = 0, usPromo = 0, themPromo = 0;
  uint32_t usBoard = 0, themBoard = 0;
  int kus = -1;
  for (int sq = 0; sq < 81; ++sq) {
    ShogiPiece cell = board.at(sq);
    if (!cell.present) continue;
    bool promo = cell.kind >= kProPawn;
    if (cell.color == us) {
      usBoard++;
      if (cell.kind == kKing) kus = sq;
      if (promo) usPromo++;
    } else {
      themBoard++;
      if (promo) themPromo++;
    }
  }
  for (int t = 0; t < 7; ++t) {
    usHand += board.hand(us, t);
    themHand += board.hand(us ^ 1, t);
  }
  bool check = false;
  if (kus >= 0) {
    for (int asq = 0; asq < 81; ++asq) {
      ShogiPiece a = board.at(asq);
      if (!a.present || a.color == us) continue;
      if (shogiAttacks(board, asq, kus)) {
        check = true;
        break;
      }
    }
  }
  float adv = 0.0f;
  if (kus >= 0) {
    int kr = sqRank(kus);
    adv = (us == kBlack) ? (8 - kr) / 8.0f : kr / 8.0f;
  }
  float lead = static_cast<float>(usBoard + usHand) - static_cast<float>(themBoard + themHand);
  int ph = phaseOf(usHand + themHand, usPromo + themPromo);
  out[0] = clamp01(static_cast<float>(us));
  out[1] = clamp01(ph / 2.0f);
  out[2] = clamp01(usHand / 20.0f);
  out[3] = clamp01(themHand / 20.0f);
  out[4] = clamp01(usPromo / 8.0f);
  out[5] = clamp01(themPromo / 8.0f);
  out[6] = clamp01(usBoard / 20.0f);
  out[7] = clamp01(themBoard / 20.0f);
  out[8] = clamp01(check ? 1.0f : 0.0f);
  out[9] = clamp01(adv);
  out[10] = clamp01((lead + 20.0f) / 40.0f);
  out[11] = clamp01(board.moveNo() / 200.0f);
}

int ShogiFeatureSet::phase(const ShogiBoard& board) {
  return phaseOf(board.handTotal(), board.promoCount());
}

}
}
