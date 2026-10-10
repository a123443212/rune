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

#include "core/xiangqi/xiangqi_moves.h"

namespace rune {
namespace xiangqi {

namespace {

bool crossedRiver(int r, uint8_t color) {
  if (color == kRed) return r >= 5;
  return r <= 4;
}

bool inPalace(int f, int r, uint8_t color) {
  if (f < 3 || f > 5) return false;
  if (color == kRed) return r >= 0 && r <= 2;
  return r >= 7 && r <= 9;
}

void pawnDests(const XiangqiBoard& board, int sq, uint8_t color, std::vector<int>& out) {
  (void)board;
  int f = sqFile(sq);
  int r = sqRank(sq);
  int fwd = (color == kRed) ? 1 : -1;
  if (onBoard(f, r + fwd)) out.push_back(makeSq(f, r + fwd));
  if (crossedRiver(r, color)) {
    if (onBoard(f - 1, r)) out.push_back(makeSq(f - 1, r));
    if (onBoard(f + 1, r)) out.push_back(makeSq(f + 1, r));
  }
}

void horseDests(const XiangqiBoard& board, int sq, std::vector<int>& out) {
  int f = sqFile(sq);
  int r = sqRank(sq);
  const int df[8] = {1, 2, 2, 1, -1, -2, -2, -1};
  const int dr[8] = {2, 1, -1, -2, -2, -1, 1, 2};
  for (int k = 0; k < 8; ++k) {
    int tf = f + df[k];
    int tr = r + dr[k];
    if (!onBoard(tf, tr)) continue;
    int a = (df[k] < 0) ? -df[k] : df[k];
    int leg = (a == 2) ? makeSq(f + df[k] / 2, r) : makeSq(f, r + dr[k] / 2);
    if (board.at(leg).present) continue;
    out.push_back(makeSq(tf, tr));
  }
}

void rayDests(const XiangqiBoard& board, int sq, int df, int dr, bool capture,
              std::vector<int>& out) {
  int f = sqFile(sq) + df;
  int r = sqRank(sq) + dr;
  while (onBoard(f, r)) {
    int t = makeSq(f, r);
    if (!board.at(t).present) {
      if (!capture) out.push_back(t);
    } else {
      if (capture) out.push_back(t);
      break;
    }
    f += df;
    r += dr;
  }
}

void rookDests(const XiangqiBoard& board, int sq, uint8_t color, std::vector<int>& out) {
  const int df[4] = {-1, 1, 0, 0};
  const int dr[4] = {0, 0, -1, 1};
  for (int k = 0; k < 4; ++k) {
    std::vector<int> quiet;
    std::vector<int> caps;
    rayDests(board, sq, df[k], dr[k], false, quiet);
    rayDests(board, sq, df[k], dr[k], true, caps);
    for (int t : quiet) out.push_back(t);
    for (int t : caps) {
      if (board.at(t).present && board.at(t).color != color) out.push_back(t);
    }
  }
}

void cannonDests(const XiangqiBoard& board, int sq, uint8_t color, std::vector<int>& out) {
  int f = sqFile(sq);
  int r = sqRank(sq);
  const int df[4] = {-1, 1, 0, 0};
  const int dr[4] = {0, 0, -1, 1};
  for (int k = 0; k < 4; ++k) {
    int cf = f + df[k];
    int cr = r + dr[k];
    while (onBoard(cf, cr) && !board.at(makeSq(cf, cr)).present) {
      out.push_back(makeSq(cf, cr));
      cf += df[k];
      cr += dr[k];
    }
    if (!onBoard(cf, cr)) continue;
    cf += df[k];
    cr += dr[k];
    while (onBoard(cf, cr)) {
      int t = makeSq(cf, cr);
      if (board.at(t).present) {
        if (board.at(t).color != color) out.push_back(t);
        break;
      }
      cf += df[k];
      cr += dr[k];
    }
  }
}

void advisorDests(int sq, uint8_t color, std::vector<int>& out) {
  int f = sqFile(sq);
  int r = sqRank(sq);
  const int df[4] = {-1, 1, -1, 1};
  const int dr[4] = {-1, -1, 1, 1};
  for (int k = 0; k < 4; ++k) {
    if (inPalace(f + df[k], r + dr[k], color)) out.push_back(makeSq(f + df[k], r + dr[k]));
  }
}

void elephantDests(const XiangqiBoard& board, int sq, uint8_t color, std::vector<int>& out) {
  int f = sqFile(sq);
  int r = sqRank(sq);
  const int df[4] = {-2, 2, -2, 2};
  const int dr[4] = {-2, -2, 2, 2};
  for (int k = 0; k < 4; ++k) {
    int tf = f + df[k];
    int tr = r + dr[k];
    if (!onBoard(tf, tr)) continue;
    if (color == kRed && tr > 4) continue;
    if (color == kBlack && tr < 5) continue;
    if (board.at(makeSq(f + df[k] / 2, r + dr[k] / 2)).present) continue;
    out.push_back(makeSq(tf, tr));
  }
}

void kingDests(const XiangqiBoard& board, int sq, uint8_t color, std::vector<int>& out) {
  int f = sqFile(sq);
  int r = sqRank(sq);
  const int df[4] = {-1, 1, 0, 0};
  const int dr[4] = {0, 0, -1, 1};
  for (int k = 0; k < 4; ++k) {
    if (inPalace(f + df[k], r + dr[k], color)) out.push_back(makeSq(f + df[k], r + dr[k]));
  }
}

}

void xiangqiPseudoDests(const XiangqiBoard& board, int sq, std::vector<int>& out) {
  out.clear();
  XiangqiPiece p = board.at(sq);
  if (!p.present) return;
  switch (p.kind) {
    case kPawn:
      pawnDests(board, sq, p.color, out);
      break;
    case kHorse:
      horseDests(board, sq, out);
      break;
    case kRook:
      rookDests(board, sq, p.color, out);
      break;
    case kCannon:
      cannonDests(board, sq, p.color, out);
      break;
    case kAdvisor:
      advisorDests(sq, p.color, out);
      break;
    case kElephant:
      elephantDests(board, sq, p.color, out);
      break;
    case kKing:
      kingDests(board, sq, p.color, out);
      break;
    default:
      break;
  }
}

}
}
