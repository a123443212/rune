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

#include "core/go/go_scratch.h"

namespace rune {
namespace go {

void GoResnetScratch::ensure(int board, int channels, int h2, int policy) {
  size_t conv = static_cast<size_t>(channels) * board * board;
  if (a.size() < conv) {
    a.assign(conv, 0.0f);
    b.assign(conv, 0.0f);
    c.assign(conv, 0.0f);
  }
  if (pooled.size() < static_cast<size_t>(channels)) pooled.assign(channels, 0.0f);
  if (h.size() < static_cast<size_t>(h2)) h.assign(h2, 0.0f);
  if (logits.size() < static_cast<size_t>(policy)) logits.assign(policy, 0.0f);
}

}
}
