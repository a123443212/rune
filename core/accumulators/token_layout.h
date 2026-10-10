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

namespace rune {

struct TokenSource {
  uint8_t group = 0;
  uint16_t lo = 0;
  uint16_t hi = 0;
};

struct TokenLayout {
  int tokens = 8;
  int dim = 32;
  std::vector<std::vector<TokenSource>> sources;

  static bool make(int tokens, int dim, TokenLayout& out, std::string& err);
  int findToken(uint8_t group, uint16_t index) const;
};

}
