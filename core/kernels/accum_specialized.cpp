#include "core/kernels/accum_specialized.h"
namespace rune {
namespace aspec {
GroupOffsets offsetsFor(int dim) {
  GroupOffsets g;
  g.dim = dim;
  for (int i = 0; i < 8; ++i) g.off[i] = i * dim;
  return g;
}
void refreshGrouped(float* acc, const EmbeddingTables* t, const std::vector<ActiveFeature>& feats, const GroupOffsets& g) {
  for (int i = 0; i < 8 * g.dim; ++i) acc[i] = 0.0f;
  applyGrouped(acc, t, feats, std::vector<ActiveFeature>{}, g);
}
void applyGrouped(float* acc, const EmbeddingTables* t, const std::vector<ActiveFeature>& added, const std::vector<ActiveFeature>& removed, const GroupOffsets& g) {
  uint16_t addIdx[8][256];
  uint16_t rmIdx[8][256];
  int addN[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int rmN[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (const ActiveFeature& f : added) {
    if (addN[f.group] < 256) addIdx[f.group][addN[f.group]++] = (uint16_t)f.index;
  }
  for (const ActiveFeature& f : removed) {
    if (rmN[f.group] < 256) rmIdx[f.group][rmN[f.group]++] = (uint16_t)f.index;
  }
  for (int grp = 0; grp < 8; ++grp) {
    if (addN[grp] == 0 && rmN[grp] == 0) continue;
    float* dst = acc + g.off[grp];
    for (int k = 0; k < addN[grp]; ++k) {
      int idx = addIdx[grp][k];
      for (int d = 0; d < g.dim; ++d) dst[d] += t->get(grp, idx, d);
    }
    for (int k = 0; k < rmN[grp]; ++k) {
      int idx = rmIdx[grp][k];
      for (int d = 0; d < g.dim; ++d) dst[d] -= t->get(grp, idx, d);
    }
  }
}
int packIds(const std::vector<ActiveFeature>& feats, uint8_t* groups, uint16_t* idx, int cap) {
  int n = (int)feats.size() < cap ? (int)feats.size() : cap;
  for (int i = 0; i < n; ++i) {
    groups[i] = feats[i].group;
    idx[i] = (uint16_t)feats[i].index;
  }
  return n;
}
}
}
