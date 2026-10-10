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

#include "core/shogi/shogi_moves.h"

namespace rune {
namespace shogi {

namespace {

bool promotable(uint8_t kind) {
  return kind == kPawn || kind == kLance || kind == kKnight || kind == kSilver ||
         kind == kBishop || kind == kRook;
}

bool inZone(int rank, uint8_t color) {
  if (color == kBlack) return rank <= 2;
  return rank >= 6;
}

bool mustPromote(uint8_t kind, uint8_t color, int toRank) {
  int last = (color == kBlack) ? 0 : 8;
  if (kind == kPawn || kind == kLance) return toRank == last;
  if (kind == kKnight) {
    if (color == kBlack) return toRank <= 1;
    return toRank >= 7;
  }
  return false;
}

void stepDests(const ShogiBoard& board, int sq, std::vector<int>& out) {
  ShogiPiece p = board.at(sq);
  int f = sqFile(sq);
  int r = sqRank(sq);
  uint8_t color = p.color;
  uint8_t kind = p.kind;
  int fwd = (color == kBlack) ? -1 : 1;
  auto step = [&](int df, int dr) {
    if (onBoard(f + df, r + dr)) out.push_back(makeSq(f + df, r + dr));
  };
  switch (kind) {
    case kPawn:
      step(0, fwd);
      break;
    case kKnight:
      step(-1, 2 * fwd);
      step(1, 2 * fwd);
      break;
    case kSilver:
      step(0, fwd);
      step(-1, fwd);
      step(1, fwd);
      step(-1, -fwd);
      step(1, -fwd);
      break;
    case kGold:
    case kProPawn:
    case kProLance:
    case kProKnight:
    case kProSilver:
      step(0, fwd);
      step(-1, fwd);
      step(1, fwd);
      step(-1, 0);
      step(1, 0);
      step(0, -fwd);
      break;
    case kKing:
      step(-1, -1);
      step(0, -1);
      step(1, -1);
      step(-1, 0);
      step(1, 0);
      step(-1, 1);
      step(0, 1);
      step(1, 1);
      break;
    case kHorse:
      step(-1, 0);
      step(1, 0);
      step(0, -1);
      step(0, 1);
      break;
    case kDragon:
      step(-1, -1);
      step(1, -1);
      step(-1, 1);
      step(1, 1);
      break;
    default:
      break;
  }
  if (kind == kLance) {
    int tr = r + fwd;
    while (onBoard(f, tr)) {
      int t = makeSq(f, tr);
      out.push_back(t);
      if (board.at(t).present) break;
      tr += fwd;
    }
  }
  if (kind == kBishop || kind == kHorse) {
    const int df[4] = {-1, 1, -1, 1};
    const int dr[4] = {-1, -1, 1, 1};
    for (int k = 0; k < 4; ++k) {
      int tf = f + df[k];
      int tr = r + dr[k];
      while (onBoard(tf, tr)) {
        int t = makeSq(tf, tr);
        out.push_back(t);
        if (board.at(t).present) break;
        tf += df[k];
        tr += dr[k];
      }
    }
  }
  if (kind == kRook || kind == kDragon) {
    const int df[4] = {-1, 1, 0, 0};
    const int dr[4] = {0, 0, -1, 1};
    for (int k = 0; k < 4; ++k) {
      int tf = f + df[k];
      int tr = r + dr[k];
      while (onBoard(tf, tr)) {
        int t = makeSq(tf, tr);
        out.push_back(t);
        if (board.at(t).present) break;
        tf += df[k];
        tr += dr[k];
      }
    }
  }
}

bool dropBlocked(const ShogiBoard& board, int to, uint8_t piece, uint8_t color) {
  int r = sqRank(to);
  int f = sqFile(to);
  if (piece == kPawn || piece == kLance) {
    if ((color == kBlack && r == 0) || (color == kWhite && r == 8)) return true;
  }
  if (piece == kKnight) {
    if ((color == kBlack && r <= 1) || (color == kWhite && r >= 7)) return true;
  }
  if (piece == kPawn) {
    for (int rr = 0; rr < 9; ++rr) {
      ShogiPiece c = board.at(makeSq(f, rr));
      if (c.present && c.kind == kPawn && c.color == color) return true;
    }
  }
  return false;
}

}

uint8_t shogiUnpromote(uint8_t kind) {
  if (kind == kProPawn) return kPawn;
  if (kind == kProLance) return kLance;
  if (kind == kProKnight) return kKnight;
  if (kind == kProSilver) return kSilver;
  if (kind == kHorse) return kBishop;
  if (kind == kDragon) return kRook;
  return kind;
}

void shogiPseudoMoves(const ShogiBoard& board, std::vector<ShogiMove>& out) {
  out.clear();
  uint8_t stm = board.sideToMove();
  std::vector<int> dests;
  for (int fr = 0; fr < 81; ++fr) {
    ShogiPiece p = board.at(fr);
    if (!p.present || p.color != stm) continue;
    dests.clear();
    stepDests(board, fr, dests);
    for (int to : dests) {
      ShogiPiece t = board.at(to);
      if (t.present && t.color == stm) continue;
      if (!promotable(p.kind)) {
        out.push_back(ShogiMove{false, fr, to, false, 0});
        continue;
      }
      bool zin = inZone(sqRank(fr), stm) || inZone(sqRank(to), stm);
      if (!zin) {
        out.push_back(ShogiMove{false, fr, to, false, 0});
        continue;
      }
      if (mustPromote(p.kind, stm, sqRank(to))) {
        out.push_back(ShogiMove{false, fr, to, true, 0});
        continue;
      }
      out.push_back(ShogiMove{false, fr, to, false, 0});
      out.push_back(ShogiMove{false, fr, to, true, 0});
    }
  }
  const uint8_t order[7] = {kPawn, kLance, kKnight, kSilver, kGold, kBishop, kRook};
  for (int i = 0; i < 7; ++i) {
    if (board.hand(stm, i) == 0) continue;
    for (int to = 0; to < 81; ++to) {
      if (board.at(to).present) continue;
      if (dropBlocked(board, to, order[i], stm)) continue;
      out.push_back(ShogiMove{true, -1, to, false, order[i]});
    }
  }
}

}
}
