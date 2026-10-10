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

#include "core/go/go_planes.h"

namespace rune {
namespace go {

void extractPlanesInto(const GoBoard& board, float* out) {
  int n = board.size();
  uint8_t stm = board.sideToMove();
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      int8_t v = board.at(r, c);
      float rel = 0.0f;
      if (v != 0) {
        bool mine = (v == 1 && stm == kBlack) || (v == -1 && stm == kWhite);
        rel = mine ? 1.0f : -1.0f;
      }
      out[r * n + c] = rel;
    }
  }
}

void extractPlanes(const GoBoard& board, std::vector<float>& out) {
  int n = board.size();
  out.assign(static_cast<size_t>(n * n), 0.0f);
  extractPlanesInto(board, out.data());
}

}
}
