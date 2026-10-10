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

#include "core/shogi/shogi_evaluator.h"

#include <stdexcept>

namespace rune {
namespace shogi {

ShogiEvaluator::ShogiEvaluator(const EmbeddingTables* tables, const IArchitecture* arch)
    : arch_(arch) {
  if (!tables) throw std::invalid_argument("null tables");
  if (!arch) throw std::invalid_argument("null arch");
  for (int g = 0; g < 9; ++g) {
    if (tables->vocab(g) < ShogiFeatureSet::vocabSize(g)) throw std::invalid_argument("tables too small");
  }
  acc_.bind(tables);
  for (int i = 0; i < 256; ++i) tokenBuf_[i] = 0.0f;
  for (int i = 0; i < 12; ++i) ctx_[i] = 0.0f;
}

bool ShogiEvaluator::refresh(const std::string& sfen) {
  ShogiBoard board;
  if (!board.setSfen(sfen)) return false;
  ShogiFeatureSet::extract(board, featureScratch_);
  acc_.refresh(featureScratch_);
  ShogiFeatureSet::context(board, ctx_);
  phase_ = ShogiFeatureSet::phase(board);
  return true;
}

void ShogiEvaluator::updateIncremental(const std::vector<ActiveFeature>& added,
                                       const std::vector<ActiveFeature>& removed) {
  acc_.applyDiff(added, removed);
}

EvalResult ShogiEvaluator::evaluate() const {
  acc_.tokens(const_cast<float*>(tokenBuf_));
  EvalResult r;
  arch_->forward(tokenBuf_, r.value, r.wdl, phase_);
  return r;
}

bool ShogiEvaluator::evaluateSfen(const std::string& sfen, EvalResult& out) {
  if (!refresh(sfen)) return false;
  out = evaluate();
  return true;
}

}
}
