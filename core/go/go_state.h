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
#include <string>
#include <vector>

#include "core/go/go_board.h"

namespace rune {
namespace go {

struct GoState {
  GoBoard board;
  bool hasKo = false;
  int ko = -1;
  float komi = 7.5f;
  uint32_t moveNo = 1;
  uint32_t passNo = 0;

  bool setState(const std::string& state);
  std::string encode() const;
  std::string normalize() const;
  void contextV02(float* out) const;
  int phaseV02() const;
};

int goKomiBucket(float komi);
int goMoveBucket(uint32_t moveNo);
int goGamePhase(uint32_t moveNo, int n);
float goClamp01(float v);

}
}
