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
#include <string>
#include <vector>
#include "core/accumulators/grouped_accumulator.h"
#include "core/architectures/base/architecture.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"
#include "core/inference/evaluator.h"
#include "core/runtime/arena.h"
namespace rune {
namespace rt {
class CompiledEvaluator {
 public:
  CompiledEvaluator(const EmbeddingTables* tables, const IArchitecture* arch, const std::string& isa, size_t arenaBytes);
  void refresh(const Board& board);
  void updateIncremental(const std::vector<ActiveFeature>& added, const std::vector<ActiveFeature>& removed);
  EvalResult evaluate();
  EvalResult evaluateBoard(const Board& board);
  const std::string& targetIsa() const;
  size_t arenaBytes() const;
 private:
  const IArchitecture* arch_;
  GroupedAccumulator acc_;
  mutable Arena arena_;
  std::string isa_;
  mutable std::vector<float> tok_;
};
}
}
