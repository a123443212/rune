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

#pragma once

#include <cstdint>
#include <vector>

#include "core/shogi/shogi_board.h"

namespace rune {
namespace shogi {

struct ShogiMove {
  bool drop = false;
  int fr = -1;
  int to = -1;
  bool promo = false;
  uint8_t piece = 0;
};

uint8_t shogiUnpromote(uint8_t kind);
void shogiPseudoMoves(const ShogiBoard& board, std::vector<ShogiMove>& out);

}
}
