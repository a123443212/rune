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
#include <cstddef>
#include <vector>
namespace rune {
namespace rt {
class Arena {
 public:
  explicit Arena(size_t bytes = 8192);
  float* at(size_t offBytes, size_t elems);
  const float* at(size_t offBytes, size_t elems) const;
  size_t bytes() const;
  void clear();
 private:
  std::vector<float> buf_;
};
size_t arenaBytesFor(int tokens, int dim, int h1, int h2);
}
}
