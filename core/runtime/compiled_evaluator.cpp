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

#include "core/runtime/compiled_evaluator.h"
#include "core/kernels/accum_specialized.h"
#include "core/kernels/fused.h"
#include "core/kernels/smallmat.h"
namespace rune {
namespace rt {
CompiledEvaluator::CompiledEvaluator(const EmbeddingTables* tables, const IArchitecture* arch, const std::string& isa, size_t arenaBytes) : arch_(arch), isa_(isa), arena_(arenaBytes == 0 ? 8192 : arenaBytes) {
  acc_.bind(tables);
  tok_.assign(256, 0.0f);
}
void CompiledEvaluator::refresh(const Board& board) {
  std::vector<ActiveFeature> feats;
  GroupedFeatureSet::extract(board, feats);
  aspec::GroupOffsets g = aspec::offsetsFor(32);
  float tmp[256];
  for (int i = 0; i < 256; ++i) tmp[i] = 0.0f;
  acc_.refresh(feats);
}
void CompiledEvaluator::updateIncremental(const std::vector<ActiveFeature>& added, const std::vector<ActiveFeature>& removed) {
  acc_.applyDiff(added, removed);
}
EvalResult CompiledEvaluator::evaluate() {
  float* tok = arena_.at(0, 256);
  acc_.tokens(tok);
  EvalResult r;
  arch_->forward(tok, r.value, r.wdl);
  return r;
}
EvalResult CompiledEvaluator::evaluateBoard(const Board& board) {
  refresh(board);
  return evaluate();
}
const std::string& CompiledEvaluator::targetIsa() const {
  return isa_;
}
size_t CompiledEvaluator::arenaBytes() const {
  return arena_.bytes();
}
}
}
