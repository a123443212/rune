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

#include "core/inference/evaluator.h"

namespace rune {

Evaluator::Evaluator(const EmbeddingTables* tables, const IArchitecture* arch) : arch_(arch) {
  acc_.bind(tables);
  for (int i = 0; i < 256; ++i) tokenBuf_[i] = 0.0f;
}

void Evaluator::refresh(const Board& board) {
  GroupedFeatureSet::extract(board, featureScratch_);
  acc_.refresh(featureScratch_);
  phase_ = board.gamePhase();
}

void Evaluator::updateIncremental(const std::vector<ActiveFeature>& added,
                                  const std::vector<ActiveFeature>& removed) {
  acc_.applyDiff(added, removed);
}

EvalResult Evaluator::evaluate() const {
  acc_.tokens(const_cast<float*>(tokenBuf_));
  EvalResult r;
  arch_->forward(tokenBuf_, r.value, r.wdl, phase_);
  return r;
}

EvalResult Evaluator::evaluateBoard(const Board& board) {
  refresh(board);
  return evaluate();
}

}
