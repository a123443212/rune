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
namespace go {

struct GoResnetScratch {
  std::vector<float> a;
  std::vector<float> b;
  std::vector<float> c;
  std::vector<float> pooled;
  std::vector<float> h;
  std::vector<float> logits;

  void ensure(int board, int channels, int h2, int policy);
};

}
}
