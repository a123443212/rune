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
  float tokenBuf_[8 * 32];
  float ctx_[12];
  int phase_ = 0;
};

}
}
