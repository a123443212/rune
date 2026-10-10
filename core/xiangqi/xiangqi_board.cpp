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

#include "core/xiangqi/xiangqi_board.h"

#include <vector>

namespace rune {
namespace xiangqi {

namespace {

int boardKind(char c) {
  switch (c) {
    case 'p': return kPawn;
    case 'h': return kHorse;
    case 'r': return kRook;
    case 'c': return kCannon;
    case 'a': return kAdvisor;
    case 'e': return kElephant;
    case 'k': return kKing;
    default: return -1;
  }
}

bool inPalace(int f, int r, uint8_t color) {
  if (f < 3 || f > 5) return false;
  if (color == kRed) return r >= 0 && r <= 2;
  return r >= 7 && r <= 9;
}

bool crossed(int r, uint8_t color) {
  if (color == kRed) return r >= 5;
  return r <= 4;
}

std::vector<std::string> splitWs(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == ' ' || c == '\t') {
      if (!cur.empty()) {
        out.push_back(cur);
        cur.clear();
      }
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

}  // namespace

XiangqiBoard::XiangqiBoard() {
  for (auto& p : squares_) p = XiangqiPiece();
}

XiangqiBoard::XiangqiBoard(const std::string& fen) : XiangqiBoard() { setFen(fen); }

bool XiangqiBoard::setFen(const std::string& fen) {
  for (auto& p : squares_) p = XiangqiPiece();
  side_ = kRed;
  moveNo_ = 1;
  std::vector<std::string> parts = splitWs(fen);
  if (parts.empty()) return false;
  std::vector<std::string> ranks;
  std::string cur;
  for (char c : parts[0]) {
    if (c == '/') {
      ranks.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  ranks.push_back(cur);
  if (ranks.size() != 10) return false;
  for (int ri = 0; ri < 10; ++ri) {
    int r = 9 - ri;
    int f = 0;
    for (char c : ranks[static_cast<size_t>(ri)]) {
      if (c >= '0' && c <= '9') {
        f += c - '0';
      } else {
        if (c < 'A' || (c > 'Z' && c < 'a') || c > 'z') return false;
        uint8_t color = (c <= 'Z') ? kRed : kBlack;
        int kind = boardKind(static_cast<char>(c | 32));
        if (kind < 0 || f >= 9) return false;
        XiangqiPiece p;
        p.kind = static_cast<uint8_t>(kind);
        p.color = color;
        p.present = true;
        squares_[static_cast<size_t>(r * 9 + f)] = p;
        f++;
      }
    }
    if (f != 9) return false;
  }
  if (parts.size() > 1) {
    if (parts[1] == "w" || parts[1] == "r") side_ = kRed;
    else if (parts[1] == "b") side_ = kBlack;
    else return false;
  }
  if (parts.size() > 5) {
    int m = 0;
    for (char c : parts[5]) {
      if (c < '0' || c > '9') break;
      m = m * 10 + (c - '0');
    }
    if (m >= 1) moveNo_ = static_cast<uint32_t>(m);
  }
  return true;
}

uint32_t XiangqiBoard::pieceCount() const {
  uint32_t n = 0;
  for (const auto& p : squares_) {
    if (p.present) n++;
  }
  return n;
}

bool flyingGenerals(const XiangqiBoard& board) {
  int ka = -1, kb = -1;
  for (int sq = 0; sq < 90; ++sq) {
    XiangqiPiece p = board.at(sq);
    if (p.present && p.kind == kKing) {
      if (p.color == kRed) ka = sq;
      else kb = sq;
    }
  }
  if (ka < 0 || kb < 0) return false;
  if (sqFile(ka) != sqFile(kb)) return false;
  int lo = ka < kb ? ka : kb;
  int hi = ka < kb ? kb : ka;
  for (int sq = lo + 9; sq < hi; sq += 9) {
    if (board.at(sq).present) return false;
  }
  return true;
}

bool xiangqiAttacks(const XiangqiBoard& board, int from, int to) {
  if (from == to) return false;
  XiangqiPiece pc = board.at(from);
  if (!pc.present) return false;
  int ff = sqFile(from), rf = sqRank(from);
  int tf = sqFile(to), rt = sqRank(to);
  int df = tf - ff, dr = rt - rf;
  int adf = df < 0 ? -df : df, adr = dr < 0 ? -dr : dr;
  int fwd = (pc.color == kRed) ? 1 : -1;
  switch (pc.kind) {
    case kKing: {
      if (adf + adr == 1 && inPalace(tf, rt, pc.color)) return true;
      XiangqiPiece vt = board.at(to);
      if (vt.present && vt.kind == kKing && tf == ff) {
        int lo = from < to ? from : to;
        int hi = from < to ? to : from;
        for (int sq = lo + 9; sq < hi; sq += 9) {
          if (board.at(sq).present) return false;
        }
        return true;
      }
      return false;
    }
    case kAdvisor:
      return adf == 1 && adr == 1 && inPalace(tf, rt, pc.color);
    case kElephant: {
      if (adf != 2 || adr != 2) return false;
      if (pc.color == kRed && rt > 4) return false;
      if (pc.color == kBlack && rt < 5) return false;
      return !board.at(makeSq(ff + df / 2, rf + dr / 2)).present;
    }
    case kHorse: {
      if (!((adf == 1 && adr == 2) || (adf == 2 && adr == 1))) return false;
      int leg = (adf == 2) ? makeSq(ff + df / 2, rf) : makeSq(ff, rf + dr / 2);
      return !board.at(leg).present;
    }
    case kRook: {
      if (df != 0 && dr != 0) return false;
      int sf = (df == 0) ? 0 : (df > 0 ? 1 : -1);
      int sr = (dr == 0) ? 0 : (dr > 0 ? 1 : -1);
      int f = ff + sf, r = rf + sr;
      while (f != tf || r != rt) {
        if (board.at(makeSq(f, r)).present) return false;
        f += sf;
        r += sr;
      }
      return true;
    }
    case kCannon: {
      if (df != 0 && dr != 0) return false;
      int sf = (df == 0) ? 0 : (df > 0 ? 1 : -1);
      int sr = (dr == 0) ? 0 : (dr > 0 ? 1 : -1);
      int f = ff + sf, r = rf + sr;
      int screens = 0;
      while (f != tf || r != rt) {
        if (board.at(makeSq(f, r)).present) screens++;
        f += sf;
        r += sr;
      }
      return screens == 1;
    }
    case kPawn: {
      if (df == 0 && dr == fwd) return true;
      if (dr == 0 && adf == 1 && crossed(rf, pc.color)) return true;
      return false;
    }
    default: return false;
  }
}

}
}
