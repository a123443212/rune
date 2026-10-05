#include "core/incremental/token_delta.h"

#include <cmath>

namespace rune {
namespace v13 {

GroupDelta detectChangedGroups(const std::vector<ActiveFeature>& before,
                               const std::vector<ActiveFeature>& after) {
  GroupDelta out;
  size_t i = 0, j = 0;
  auto key = [](const ActiveFeature& f) {
    return (static_cast<uint32_t>(f.group) << 16) | f.index;
  };
  std::vector<ActiveFeature> addedAll;
  std::vector<ActiveFeature> removedAll;
  while (i < before.size() && j < after.size()) {
    uint32_t kb = key(before[i]);
    uint32_t ka = key(after[j]);
    if (kb == ka) {
      ++i;
      ++j;
    } else if (kb < ka) {
      removedAll.push_back(before[i++]);
    } else {
      addedAll.push_back(after[j++]);
    }
  }
  while (i < before.size()) removedAll.push_back(before[i++]);
  while (j < after.size()) addedAll.push_back(after[j++]);
  bool touched[GroupedFeatureSet::kNumGroups] = {};
  for (const auto& f : addedAll) {
    out.added[f.group].push_back(f);
    touched[f.group] = true;
  }
  for (const auto& f : removedAll) {
    out.removed[f.group].push_back(f);
    touched[f.group] = true;
  }
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) {
    if (touched[g]) out.changedGroups.push_back(g);
  }
  return out;
}

std::vector<int> changedTokensFromGroups(const GroupDelta& delta,
                                         const int* tokenOfGroup, int numTokens) {
  std::vector<int> seen(static_cast<size_t>(numTokens), 0);
  for (int g : delta.changedGroups) {
    int t = tokenOfGroup[g];
    if (t >= 0 && t < numTokens) seen[static_cast<size_t>(t)] = 1;
  }
  std::vector<int> out;
  for (int t = 0; t < numTokens; ++t) {
    if (seen[static_cast<size_t>(t)]) out.push_back(t);
  }
  return out;
}

std::vector<int> changedTokensByCompare(const float* tokOld, const float* tokNew,
                                        int numTokens, int dim) {
  std::vector<int> out;
  for (int t = 0; t < numTokens; ++t) {
    bool same = true;
    for (int d = 0; d < dim; ++d) {
      if (tokOld[t * dim + d] != tokNew[t * dim + d]) {
        same = false;
        break;
      }
    }
    if (!same) out.push_back(t);
  }
  return out;
}

void applyAccumDelta(const EmbeddingTables& tables, float* acc, int numGroups, int dim,
                     const GroupDelta& delta) {
  for (int g = 0; g < numGroups; ++g) {
    for (const auto& f : delta.added[g]) {
      for (int d = 0; d < dim; ++d) acc[g * dim + d] += tables.get(f.group, f.index, d);
    }
    for (const auto& f : delta.removed[g]) {
      for (int d = 0; d < dim; ++d) acc[g * dim + d] -= tables.get(f.group, f.index, d);
    }
  }
}

}
}
