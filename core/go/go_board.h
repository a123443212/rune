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
namespace go {

constexpr uint8_t kBlack = 0;
constexpr uint8_t kWhite = 1;

class GoBoard {
 public:
  GoBoard();
  GoBoard(int size, const std::string& state);

  bool setState(const std::string& state);
  bool setState(int size, const std::string& state);

  int size() const { return size_; }
  int8_t at(int r, int c) const { return stones_[static_cast<size_t>(r * size_ + c)]; }
  int8_t atSq(int sq) const { return stones_[static_cast<size_t>(sq)]; }
  uint8_t sideToMove() const { return side_; }
  uint32_t stoneCount(int8_t v) const;

  int libertiesOf(int r, int c) const;

 private:
  int size_ = 9;
  std::vector<int8_t> stones_;
  uint8_t side_ = kBlack;
};

bool validGoSize(int n);

}
}
