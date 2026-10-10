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

#include "core/go/go_token_v02.h"

#include <algorithm>

#include "core/go/go_rules.h"
#include "core/go/go_state.h"

namespace rune {
namespace go {

const char* GoTokenV02::version() { return "go_planes_v02"; }

void GoTokenV02::extract(const GoState& st, std::vector<ActiveFeature>& out) {
  out.clear();
  int n = st.board.size();
  uint8_t stm = st.board.sideToMove();
  std::vector<int8_t> raw(static_cast<size_t>(n * n));
  for (int sq = 0; sq < n * n; ++sq) raw[static_cast<size_t>(sq)] = st.board.atSq(sq);
  std::vector<int> libs;
  goLibertyMap(raw, n, libs);
  int atari = 0;
  for (int sq = 0; sq < n * n; ++sq) {
    int8_t v = raw[static_cast<size_t>(sq)];
    if (v == 0) continue;
    int ci = ((v == 1 && stm == kBlack) || (v == -1 && stm == kWhite)) ? 0 : 1;
    uint16_t key = static_cast<uint16_t>((ci * 181 + sq) % 361);
    ActiveFeature f;
    f.group = 0;
    f.index = key;
    out.push_back(f);
    f.group = 6;
    f.index = static_cast<uint16_t>(sq % 361);
    out.push_back(f);
    int k = libs[static_cast<size_t>(sq)];
    if (k <= 2) {
      f.group = 1;
      f.index = key;
      out.push_back(f);
    }
    if (k == 1) {
      f.group = 5;
      f.index = static_cast<uint16_t>(sq % 361);
      out.push_back(f);
      ++atari;
    }
  }
  ActiveFeature g;
  if (st.hasKo) {
    g.group = 2;
    g.index = static_cast<uint16_t>(st.ko % 361);
    out.push_back(g);
    g.group = 5;
    g.index = static_cast<uint16_t>(360 - (st.ko % 361));
    out.push_back(g);
  }
  g.group = 7;
  g.index = static_cast<uint16_t>(stm);
  out.push_back(g);
  g.index = static_cast<uint16_t>(2 + goKomiBucket(st.komi));
  out.push_back(g);
  g.index = static_cast<uint16_t>(10 + goMoveBucket(st.moveNo));
  out.push_back(g);
  uint32_t pn = st.passNo > 2 ? 2 : st.passNo;
  g.index = static_cast<uint16_t>(16 + pn);
  out.push_back(g);
  if (st.hasKo) {
    g.index = 19;
    out.push_back(g);
  }
  int mb = goMoveBucket(st.moveNo);
  int ph = (mb <= 1) ? 0 : ((mb <= 3) ? 1 : 2);
  g.index = static_cast<uint16_t>(20 + ph);
  out.push_back(g);
  g.group = 8;
  g.index = static_cast<uint16_t>(atari > 8 ? 8 : atari);
  out.push_back(g);
  if (st.hasKo) {
    g.index = static_cast<uint16_t>(9 + (st.ko % 9));
  } else {
    g.index = 9;
  }
  out.push_back(g);
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
}

int GoTokenV02::phaseFromGroup7(const std::vector<ActiveFeature>& feats) {
  int best = -1;
  for (const auto& f : feats) {
    if (f.group == 7 && f.index >= 20 && f.index <= 22) {
      if (static_cast<int>(f.index) > best) best = f.index;
    }
  }
  if (best < 0) return 0;
  return best - 20;
}

}
}
