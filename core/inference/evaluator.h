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

#include <vector>

#include "core/accumulators/grouped_accumulator.h"
#include "core/architectures/base/architecture.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"

namespace rune {

struct EvalResult {
  float value = 0.0f;
  float wdl[3] = {0.0f, 0.0f, 0.0f};
};

class Evaluator {
 public:
  Evaluator(const EmbeddingTables* tables, const IArchitecture* arch);

  void refresh(const Board& board);
  void updateIncremental(const std::vector<ActiveFeature>& added,
                         const std::vector<ActiveFeature>& removed);
  EvalResult evaluate() const;
  EvalResult evaluateBoard(const Board& board);

  void currentTokens(float* out) const { acc_.tokens(out); }

 private:
  const IArchitecture* arch_;
  GroupedAccumulator acc_;
  std::vector<ActiveFeature> featureScratch_;
  float tokenBuf_[8 * 32];
  int phase_ = 1;
};

}
