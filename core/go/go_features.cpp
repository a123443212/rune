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

#include "core/go/go_features.h"

#include <algorithm>

namespace rune {
namespace go {

namespace {

const int kVocabs[9] = {722, 722, 361, 64, 64, 361, 361, 64, 18};

float clamp01(float v) {
  if (v < 0.0f) return 0.0f;
  if (v > 1.0f) return 1.0f;
  return v;
}

}

int GoFeatureSet::vocabSize(int group) { return kVocabs[group]; }

const char* GoFeatureSet::version() { return "go_planes_v01"; }

void GoFeatureSet::extract(const GoBoard& board, std::vector<ActiveFeature>& out) {
  out.clear();
  int n = board.size();
  uint8_t stm = board.sideToMove();
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      int8_t v = board.at(r, c);
      if (v == 0) continue;
      bool mine = (v == 1 && stm == kBlack) || (v == -1 && stm == kWhite);
      int ci = mine ? 0 : 1;
      int sq = r * n + c;
      ActiveFeature f;
      f.group = 0;
      f.index = static_cast<uint16_t>(ci * 361 + sq);
      out.push_back(f);
      f.group = 6;
      f.index = static_cast<uint16_t>(sq % 361);
      out.push_back(f);
    }
  }
  ActiveFeature g;
  g.group = 7;
  g.index = stm;
  out.push_back(g);
  g.index = 2;
  out.push_back(g);
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
}

void GoFeatureSet::context(const GoBoard& board, float* out) {
  int n = board.size();
  uint8_t stm = board.sideToMove();
  float us = 0.0f, them = 0.0f, empty = 0.0f;
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      int8_t v = board.at(r, c);
      if (v == 0) {
        empty += 1.0f;
      } else {
        bool mine = (v == 1 && stm == kBlack) || (v == -1 && stm == kWhite);
        if (mine) us += 1.0f;
        else them += 1.0f;
      }
    }
  }
  float total = static_cast<float>(n * n);
  out[0] = clamp01(static_cast<float>(stm));
  out[1] = clamp01(us / total);
  out[2] = clamp01(them / total);
  out[3] = 0.5f;
  out[4] = 0.0f;
  out[5] = 0.0f;
  out[6] = 0.0f;
  out[7] = 0.0f;
  out[8] = 0.0f;
  out[9] = 0.0f;
  out[10] = clamp01(empty / total);
  out[11] = 0.0f;
}

int GoFeatureSet::phase(const GoBoard& board) {
  (void)board;
  return 0;
}

}
}
