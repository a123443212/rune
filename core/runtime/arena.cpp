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

#include "core/runtime/arena.h"
#include <stdexcept>
namespace rune {
namespace rt {
Arena::Arena(size_t bytes) {
  size_t elems = (bytes + 3) / 4;
  if (elems == 0) elems = 1;
  buf_.assign(elems, 0.0f);
}
float* Arena::at(size_t offBytes, size_t elems) {
  size_t o = offBytes / 4;
  if (o + elems > buf_.size()) throw std::out_of_range("arena out of range");
  return buf_.data() + o;
}
const float* Arena::at(size_t offBytes, size_t elems) const {
  size_t o = offBytes / 4;
  if (o + elems > buf_.size()) throw std::out_of_range("arena out of range");
  return buf_.data() + o;
}
size_t Arena::bytes() const {
  return buf_.size() * 4;
}
void Arena::clear() {
  for (float& v : buf_) v = 0.0f;
}
size_t arenaBytesFor(int tokens, int dim, int h1, int h2) {
  size_t live[9] = {(size_t)tokens * dim, (size_t)tokens * dim, (size_t)tokens * dim, (size_t)tokens * dim, (size_t)tokens * tokens, (size_t)tokens * dim, (size_t)h1, (size_t)h2, (size_t)tokens * dim};
  size_t t = 0;
  for (size_t e : live) t += e * 4 + 32;
  return (t + 31) / 32 * 32;
}
}
}
