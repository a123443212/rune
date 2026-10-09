#include "core/xiangqi/xiangqi_evaluator.h"

namespace rune {
namespace xiangqi {

XiangqiEvaluator::XiangqiEvaluator(const EmbeddingTables* tables, const IArchitecture* arch)
    : arch_(arch) {
  acc_.bind(tables);
  for (int i = 0; i < 256; ++i) tokenBuf_[i] = 0.0f;
  for (int i = 0; i < 12; ++i) ctx_[i] = 0.0f;
}

bool XiangqiEvaluator::refresh(const std::string& fen) {
  XiangqiBoard board;
  if (!board.setFen(fen)) return false;
  std::vector<ActiveFeature> feats;
  XiangqiFeatureSet::extract(board, feats);
  acc_.refresh(feats);
  XiangqiFeatureSet::context(board, ctx_);
  phase_ = XiangqiFeatureSet::phase(board);
  return true;
}

void XiangqiEvaluator::updateIncremental(const std::vector<ActiveFeature>& added,
                                         const std::vector<ActiveFeature>& removed) {
  acc_.applyDiff(added, removed);
}

EvalResult XiangqiEvaluator::evaluate() const {
  acc_.tokens(const_cast<float*>(tokenBuf_));
  EvalResult r;
  arch_->forward(tokenBuf_, r.value, r.wdl, phase_);
  return r;
}

bool XiangqiEvaluator::evaluateFen(const std::string& fen, EvalResult& out) {
  if (!refresh(fen)) return false;
  out = evaluate();
  return true;
}

}
}
