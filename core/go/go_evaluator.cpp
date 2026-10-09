#include "core/go/go_evaluator.h"

namespace rune {
namespace go {

GoEvaluator::GoEvaluator(const EmbeddingTables* tables, const IArchitecture* arch)
    : arch_(arch) {
  acc_.bind(tables);
  for (int i = 0; i < 256; ++i) tokenBuf_[i] = 0.0f;
  for (int i = 0; i < 12; ++i) ctx_[i] = 0.0f;
}

bool GoEvaluator::refresh(const std::string& state) {
  GoBoard board;
  if (!board.setState(state)) return false;
  std::vector<ActiveFeature> feats;
  GoFeatureSet::extract(board, feats);
  acc_.refresh(feats);
  GoFeatureSet::context(board, ctx_);
  phase_ = GoFeatureSet::phase(board);
  return true;
}

void GoEvaluator::updateIncremental(const std::vector<ActiveFeature>& added,
                                    const std::vector<ActiveFeature>& removed) {
  acc_.applyDiff(added, removed);
}

EvalResult GoEvaluator::evaluate() const {
  acc_.tokens(const_cast<float*>(tokenBuf_));
  EvalResult r;
  arch_->forward(tokenBuf_, r.value, r.wdl, phase_);
  return r;
}

bool GoEvaluator::evaluateState(const std::string& state, EvalResult& out) {
  if (!refresh(state)) return false;
  out = evaluate();
  return true;
}

}
}
