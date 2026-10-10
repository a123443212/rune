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
#include "core/go/go_board.h"
#include "core/go/go_features.h"
#include "core/inference/evaluator.h"

namespace rune {
namespace go {

class GoEvaluator {
 public:
  GoEvaluator(const EmbeddingTables* tables, const IArchitecture* arch);

  bool refresh(const std::string& state);
  void updateIncremental(const std::vector<ActiveFeature>& added,
                         const std::vector<ActiveFeature>& removed);
  EvalResult evaluate() const;
  bool evaluateState(const std::string& state, EvalResult& out);

  void currentTokens(float* out) const { acc_.tokens(out); }
  void currentContext(float* out) const {
    for (int i = 0; i < GoFeatureSet::kContextDim; ++i) out[i] = ctx_[i];
  }
  int phase() const { return phase_; }

 private:
  const IArchitecture* arch_;
  GroupedAccumulator acc_;
  std::vector<ActiveFeature> featureScratch_;
  float tokenBuf_[8 * 32];
  float ctx_[12];
  int phase_ = 0;
};

}
}
