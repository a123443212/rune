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

#include "core/go/go_state.h"

#include "core/go/go_rules.h"

namespace rune {
namespace go {

namespace {

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

bool parseKomi(const std::string& t, float& v) {
  try {
    size_t pos = 0;
    v = std::stof(t, &pos);
    if (pos != t.size()) return false;
  } catch (...) {
    return false;
  }
  if (v < -50.0f || v > 50.0f) return false;
  return true;
}

bool parseKo(const std::string& t, int n, bool& has, int& ko) {
  if (t == "-") {
    has = false;
    ko = -1;
    return true;
  }
  try {
    size_t pos = 0;
    long v = std::stol(t, &pos);
    if (pos != t.size()) return false;
    if (v < 0 || v >= static_cast<long>(n) * n) return false;
    has = true;
    ko = static_cast<int>(v);
    return true;
  } catch (...) {
    return false;
  }
}

bool parseCount(const std::string& t, uint32_t lo, uint32_t hi, uint32_t& v) {
  try {
    size_t pos = 0;
    long x = std::stol(t, &pos);
    if (pos != t.size()) return false;
    if (x < static_cast<long>(lo) || x > static_cast<long>(hi)) return false;
    v = static_cast<uint32_t>(x);
    return true;
  } catch (...) {
    return false;
  }
}

}

float goClamp01(float v) {
  if (v < 0.0f) return 0.0f;
  if (v > 1.0f) return 1.0f;
  return v;
}

int goKomiBucket(float komi) {
  float v = goClamp01((komi + 5.0f) / 20.0f);
  int b = static_cast<int>(v * 7.99f);
  if (b < 0) return 0;
  if (b > 7) return 7;
  return b;
}

int goMoveBucket(uint32_t moveNo) {
  uint32_t b = (moveNo - 1) / 20;
  if (b > 5) return 5;
  return static_cast<int>(b);
}

int goGamePhase(uint32_t moveNo, int n) {
  uint32_t total = static_cast<uint32_t>(n * n);
  if (moveNo <= total / 3) return 0;
  if (moveNo <= (2 * total) / 3) return 1;
  return 2;
}

bool GoState::setState(const std::string& state) {
  std::vector<std::string> parts = splitWs(state);
  if (parts.empty()) return false;
  if (!board.setState(state)) return false;
  int n = board.size();
  hasKo = false;
  ko = -1;
  komi = 7.5f;
  moveNo = 1;
  passNo = 0;
  if (parts.size() > 2) {
    if (!parseKo(parts[2], n, hasKo, ko)) return false;
  }
  if (parts.size() > 3) {
    float kv = 7.5f;
    if (!parseKomi(parts[3], kv)) return false;
    komi = kv;
  }
  if (parts.size() > 4) {
    uint32_t mv = 1;
    if (!parseCount(parts[4], 1, 10000, mv)) return false;
    moveNo = mv;
  }
  if (parts.size() > 5) {
    uint32_t pv = 0;
    if (!parseCount(parts[5], 0, 500, pv)) return false;
    passNo = pv;
  }
  return true;
}

std::string GoState::encode() const {
  int n = board.size();
  std::string grid;
  for (int r = 0; r < n; ++r) {
    if (r) grid += "/";
    for (int c = 0; c < n; ++c) {
      int8_t v = board.at(r, c);
      if (v == 0) grid += ".";
      else if (v == 1) grid += "X";
      else grid += "O";
    }
  }
  std::string s = (board.sideToMove() == kBlack) ? "b" : "w";
  std::string k = hasKo ? std::to_string(ko) : "-";
  return grid + " " + s + " " + k + " " + std::to_string(komi) + " " + std::to_string(moveNo) +
         " " + std::to_string(passNo);
}

std::string GoState::normalize() const {
  int n = board.size();
  std::string grid;
  for (int r = 0; r < n; ++r) {
    if (r) grid += "/";
    for (int c = 0; c < n; ++c) {
      int8_t v = board.at(r, c);
      if (v == 0) grid += ".";
      else if (v == 1) grid += "X";
      else grid += "O";
    }
  }
  std::string s = (board.sideToMove() == kBlack) ? "b" : "w";
  if (hasKo) return grid + " " + s + " " + std::to_string(ko);
  return grid + " " + s;
}

void GoState::contextV02(float* out) const {
  int n = board.size();
  uint8_t stm = board.sideToMove();
  float us = 0.0f;
  float them = 0.0f;
  float empty = 0.0f;
  int total = n * n;
  for (int sq = 0; sq < total; ++sq) {
    int8_t v = board.atSq(sq);
    if (v == 0) {
      empty += 1.0f;
    } else {
      bool mine = (v == 1 && stm == kBlack) || (v == -1 && stm == kWhite);
      if (mine) us += 1.0f;
      else them += 1.0f;
    }
  }
  std::vector<int8_t> raw(static_cast<size_t>(total));
  for (int sq = 0; sq < total; ++sq) raw[static_cast<size_t>(sq)] = board.atSq(sq);
  std::vector<int> libs;
  goLibertyMap(raw, n, libs);
  float l1 = 0.0f;
  float l2 = 0.0f;
  float l3 = 0.0f;
  for (int sq = 0; sq < total; ++sq) {
    int8_t v = raw[static_cast<size_t>(sq)];
    if (v == 0) continue;
    bool mine = (v == 1 && stm == kBlack) || (v == -1 && stm == kWhite);
    if (!mine) continue;
    int k = libs[static_cast<size_t>(sq)];
    if (k <= 1) l1 += 1.0f;
    else if (k == 2) l2 += 1.0f;
    else l3 += 1.0f;
  }
  float f1 = 0.0f;
  float f2 = 0.0f;
  float f3 = 0.0f;
  if (us > 0.0f) {
    f1 = goClamp01(l1 / us);
    f2 = goClamp01(l2 / us);
    f3 = goClamp01(l3 / us);
  }
  out[0] = goClamp01(static_cast<float>(stm));
  out[1] = goClamp01(us / static_cast<float>(total));
  out[2] = goClamp01(them / static_cast<float>(total));
  out[3] = goClamp01(komi / 15.0f);
  out[4] = goClamp01(static_cast<float>(moveNo) / 200.0f);
  out[5] = goClamp01(static_cast<float>(passNo) / 2.0f);
  out[6] = hasKo ? 1.0f : 0.0f;
  out[7] = goClamp01(f1);
  out[8] = goClamp01(f2);
  out[9] = goClamp01(f3);
  out[10] = goClamp01(empty / static_cast<float>(total));
  out[11] = goClamp01(static_cast<float>(goGamePhase(moveNo, n)) / 2.0f);
}

int GoState::phaseV02() const { return goGamePhase(moveNo, board.size()); }

}
}
