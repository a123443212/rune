#include "core/shogi/shogi_evaluator.h"

namespace rune {
namespace shogi {

ShogiEvaluator::ShogiEvaluator(const EmbeddingTables* tables, const IArchitecture* arch)
    : arch_(arch) {
  acc_.bind(tables);
  for (int i = 0; i < 256; ++i) tokenBuf_[i] = 0.0f;
  for (int i = 0; i < 12; ++i) ctx_[i] = 0.0f;
}

bool ShogiEvaluator::refresh(const std::string& sfen) {
  ShogiBoard board;
  if (!board.setSfen(sfen)) return false;
  std::vector<ActiveFeature> feats;
  ShogiFeatureSet::extract(board, feats);
  acc_.refresh(feats);
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
