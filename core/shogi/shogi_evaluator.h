#pragma once

#include <string>
#include <vector>

#include "core/accumulators/grouped_accumulator.h"
#include "core/architectures/base/architecture.h"
#include "core/inference/evaluator.h"
#include "core/shogi/shogi_board.h"
#include "core/shogi/shogi_features.h"

namespace rune {
namespace shogi {

class ShogiEvaluator {
 public:
  ShogiEvaluator(const EmbeddingTables* tables, const IArchitecture* arch);

  bool refresh(const std::string& sfen);
  void updateIncremental(const std::vector<ActiveFeature>& added,
                         const std::vector<ActiveFeature>& removed);
  EvalResult evaluate() const;
  bool evaluateSfen(const std::string& sfen, EvalResult& out);

  void currentTokens(float* out) const { acc_.tokens(out); }
  void currentContext(float* out) const {
    for (int i = 0; i < ShogiFeatureSet::kContextDim; ++i) out[i] = ctx_[i];
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
