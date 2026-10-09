#include "core/inference/evaluator.h"

namespace rune {

Evaluator::Evaluator(const EmbeddingTables* tables, const IArchitecture* arch) : arch_(arch) {
  acc_.bind(tables);
  for (int i = 0; i < 256; ++i) tokenBuf_[i] = 0.0f;
}

void Evaluator::refresh(const Board& board) {
  std::vector<ActiveFeature> feats;
  GroupedFeatureSet::extract(board, feats);
  acc_.refresh(feats);
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
