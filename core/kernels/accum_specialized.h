#pragma once
#include <vector>
#include "core/features/feature_set.h"
#include "core/accumulators/grouped_accumulator.h"
namespace rune {
namespace aspec {
struct GroupOffsets {
  int dim = 32;
  int off[9] = {0, 32, 64, 96, 128, 160, 192, 224, 0};
};
GroupOffsets offsetsFor(int dim);
void refreshGrouped(float* acc, const EmbeddingTables* t, const std::vector<ActiveFeature>& feats, const GroupOffsets& g);
void applyGrouped(float* acc, const EmbeddingTables* t, const std::vector<ActiveFeature>& added, const std::vector<ActiveFeature>& removed, const GroupOffsets& g);
int packIds(const std::vector<ActiveFeature>& feats, uint8_t* groups, uint16_t* idx, int cap);
}
}
