#pragma once

#include <vector>

#include "core/incremental/relational_cache.h"

namespace rune {
namespace v13 {

void denseBaselineForward(const IncrWeights& w, const float* tokens, const float* ctx,
                          float* out);

void qkvRowsForward(const IncrWeights& w, const float* tokens,
                    const std::vector<int>& changed, float* q, float* k, float* vv);

int scoreCellsForChanged(int tokens, const std::vector<int>& changed);

}
}
