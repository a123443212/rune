#pragma once

#include <string>
#include <vector>

#include "core/accumulators/grouped_accumulator.h"
#include "core/architectures/base/architecture.h"
#include "core/inference/evaluator.h"
#include "core/xiangqi/xiangqi_board.h"
#include "core/xiangqi/xiangqi_features.h"

namespace rune {
namespace xiangqi {

class XiangqiEvaluator {
 public:
  XiangqiEvaluator(const EmbeddingTables* tables, const IArchitecture* arch);

  bool refresh(const std::string& fen);
  void updateIncremental(const std::vector<ActiveFeature>& added,
                         const std::vector<ActiveFeature>& removed);
  EvalResult evaluate() const;
  bool evaluateFen(const std::string& fen, EvalResult& out);

  void currentTokens(float* out) const { acc_.tokens(out); }
  void currentContext(float* out) const {
    for (int i = 0; i < XiangqiFeatureSet::kContextDim; ++i) out[i] = ctx_[i];
  }
  int phase() const { return phase_; }

 private:
  const IArchitecture* arch_;
  GroupedAccumulator acc_;
  float tokenBuf_[8 * 32];
  float ctx_[12];
  int phase_ = 0;
};

}
}
