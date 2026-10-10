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

#include "core/xiangqi/xiangqi_features.h"

#include <algorithm>

namespace rune {
namespace xiangqi {

namespace {

const int kVocabs[9] = {198, 180, 360, 180, 360, 639, 694, 64, 18};

int phaseOf(const XiangqiBoard& board) {
  int n = 0;
  for (int sq = 0; sq < 90; ++sq) {
    XiangqiPiece c = board.at(sq);
    if (!c.present) continue;
    if (c.kind == kAdvisor || c.kind == kElephant || c.kind == kHorse || c.kind == kRook ||
        c.kind == kCannon) {
      n++;
    }
  }
  if (n >= 14) return 0;
  if (n >= 8) return 1;
  return 2;
}

float clamp01(float v) {
  if (v < 0.0f) return 0.0f;
  if (v > 1.0f) return 1.0f;
  return v;
}

bool crossedSq(int sq, uint8_t color) {
  int r = sqRank(sq);
  if (color == kRed) return r >= 5;
  return r <= 4;
}

}  // namespace

int XiangqiFeatureSet::vocabSize(int group) { return kVocabs[group]; }

const char* XiangqiFeatureSet::version() { return "xiangqi_raw_v01"; }

void XiangqiFeatureSet::extract(const XiangqiBoard& board, std::vector<ActiveFeature>& out) {
  out.clear();
  uint8_t us = board.sideToMove();
  for (int sq = 0; sq < 90; ++sq) {
    XiangqiPiece cell = board.at(sq);
    if (!cell.present) continue;
    uint16_t ci = (cell.color == us) ? 0 : 1;
    ActiveFeature f;
    if (cell.kind == kPawn) {
      f.group = 0;
      f.index = static_cast<uint16_t>(ci * 90 + sq);
      out.push_back(f);
      if (crossedSq(sq, cell.color)) {
        f.index = static_cast<uint16_t>(180 + ci * 9 + sqFile(sq));
        out.push_back(f);
      }
    } else if (cell.kind == kKing) {
      f.group = 1;
      f.index = static_cast<uint16_t>(ci * 90 + sq);
      out.push_back(f);
    } else if (cell.kind == kAdvisor || cell.kind == kElephant) {
      f.group = 2;
      f.index = static_cast<uint16_t>((ci * 2 + (cell.kind - kAdvisor)) * 90 + sq);
      out.push_back(f);
    } else if (cell.kind == kHorse) {
      f.group = 3;
      f.index = static_cast<uint16_t>(ci * 90 + sq);
      out.push_back(f);
    } else if (cell.kind == kRook || cell.kind == kCannon) {
      f.group = 4;
      f.index = static_cast<uint16_t>((ci * 2 + (cell.kind - kRook)) * 90 + sq);
      out.push_back(f);
    }
    if (cell.kind <= kKing) {
      f.group = 6;
      f.index = static_cast<uint16_t>(cell.kind * 90 + sq);
      out.push_back(f);
    }
  }
  bool fly = flyingGenerals(board);
  for (int vsq = 0; vsq < 90; ++vsq) {
    XiangqiPiece victim = board.at(vsq);
    if (!victim.present) continue;
    for (int asq = 0; asq < 90; ++asq) {
      XiangqiPiece attacker = board.at(asq);
      if (!attacker.present || attacker.color == victim.color) continue;
      if (!xiangqiAttacks(board, asq, vsq)) continue;
      ActiveFeature f;
      f.group = 5;
      f.index = static_cast<uint16_t>(victim.kind * 90 + vsq);
      out.push_back(f);
      f.index = static_cast<uint16_t>(630 + sqFile(asq));
      out.push_back(f);
    }
  }
  for (int f = 0; f < 9; ++f) {
    int majors = 0, majorsT = 0;
    for (int r = 0; r < 10; ++r) {
      XiangqiPiece c = board.at(makeSq(f, r));
      if (!c.present) continue;
      if (c.kind == kHorse || c.kind == kRook || c.kind == kCannon) {
        if (c.color == us) majors++;
        else majorsT++;
      }
    }
    ActiveFeature b;
    b.group = 8;
    if (majors >= 2) {
      b.index = static_cast<uint16_t>(f);
      out.push_back(b);
    }
    if (majorsT >= 2) {
      b.index = static_cast<uint16_t>(9 + f);
      out.push_back(b);
    }
  }
  ActiveFeature g;
  g.group = 7;
  g.index = us;
  out.push_back(g);
  g.index = static_cast<uint16_t>(2 + phaseOf(board));
  out.push_back(g);
  uint32_t total = board.pieceCount();
  uint32_t bucket = (32 - total) / 2;
  if (bucket > 15) bucket = 15;
  g.index = static_cast<uint16_t>(5 + bucket);
  out.push_back(g);
  int kus = -1;
  for (int sq = 0; sq < 90; ++sq) {
    XiangqiPiece c = board.at(sq);
    if (c.present && c.kind == kKing && c.color == us) {
      kus = sq;
      break;
    }
  }
  bool check = false;
  if (kus >= 0) {
    for (int asq = 0; asq < 90; ++asq) {
      XiangqiPiece a = board.at(asq);
      if (!a.present || a.color == us) continue;
      if (xiangqiAttacks(board, asq, kus)) {
        check = true;
        break;
      }
    }
    if (!check && fly) check = true;
  }
  if (check) {
    g.index = 21;
    out.push_back(g);
  }
  if (fly) {
    g.index = 22;
    out.push_back(g);
  }
  uint32_t mv = board.moveNo() / 20;
  if (mv > 5) mv = 5;
  g.index = static_cast<uint16_t>(23 + mv);
  out.push_back(g);
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
}

void XiangqiFeatureSet::context(const XiangqiBoard& board, float* out) {
  uint8_t us = board.sideToMove();
  uint32_t usMajors = 0, themMajors = 0, usPawns = 0, themPawns = 0;
  uint32_t usCrossed = 0, usPalace = 0, usTotal = 0, themTotal = 0;
  int kus = -1;
  for (int sq = 0; sq < 90; ++sq) {
    XiangqiPiece cell = board.at(sq);
    if (!cell.present) continue;
    if (cell.color == us) {
      usTotal++;
      if (cell.kind == kHorse || cell.kind == kRook || cell.kind == kCannon) usMajors++;
      if (cell.kind == kPawn) {
        usPawns++;
        if (crossedSq(sq, cell.color)) usCrossed++;
      }
      if (cell.kind == kAdvisor || cell.kind == kElephant) usPalace++;
      if (cell.kind == kKing) kus = sq;
    } else {
      themTotal++;
      if (cell.kind == kHorse || cell.kind == kRook || cell.kind == kCannon) themMajors++;
      if (cell.kind == kPawn) themPawns++;
    }
  }
  bool fly = flyingGenerals(board);
  bool check = false;
  if (kus >= 0) {
    for (int asq = 0; asq < 90; ++asq) {
      XiangqiPiece a = board.at(asq);
      if (!a.present || a.color == us) continue;
      if (xiangqiAttacks(board, asq, kus)) {
        check = true;
        break;
      }
    }
    if (!check && fly) check = true;
  }
  float lead = static_cast<float>(usTotal) - static_cast<float>(themTotal);
  out[0] = clamp01(static_cast<float>(us));
  out[1] = clamp01(phaseOf(board) / 2.0f);
  out[2] = clamp01(usMajors / 12.0f);
  out[3] = clamp01(themMajors / 12.0f);
  out[4] = clamp01(usPawns / 5.0f);
  out[5] = clamp01(themPawns / 5.0f);
  out[6] = clamp01(usCrossed / 5.0f);
  out[7] = clamp01(check ? 1.0f : 0.0f);
  out[8] = clamp01(fly ? 1.0f : 0.0f);
  out[9] = clamp01(usPalace / 4.0f);
  out[10] = clamp01((lead + 16.0f) / 32.0f);
  out[11] = clamp01(board.moveNo() / 200.0f);
}

int XiangqiFeatureSet::phase(const XiangqiBoard& board) { return phaseOf(board); }

}
}
