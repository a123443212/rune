#pragma once

#include <vector>

#include "core/accumulators/grouped_accumulator.h"
#include "core/architectures/base/architecture.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"

namespace rune {

struct EvalResult {
  float value = 0.0f;
  float wdl[3] = {0.0f, 0.0f, 0.0f};
};

class Evaluator {
 public:
  Evaluator(const EmbeddingTables* tables, const IArchitecture* arch);

  void refresh(const Board& board);
  void updateIncremental(const std::vector<ActiveFeature>& added,
                         const std::vector<ActiveFeature>& removed);
  EvalResult evaluate() const;
  EvalResult evaluateBoard(const Board& board);

  void currentTokens(float* out) const { acc_.tokens(out); }

 private:
  const IArchitecture* arch_;
  GroupedAccumulator acc_;
  float tokenBuf_[8 * 32];
};

}
