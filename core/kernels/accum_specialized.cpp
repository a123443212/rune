/*
RUNE — Relational Unified Neural Evaluator
Copyright (C) 2026 a123443212

SPDX-License-Identifier: MIT OR Apache-2.0

This project is dual-licensed under the MIT License and the
Apache License, Version 2.0. You may choose either license
when using, copying, modifying, or distributing this software.

MIT License: https://opensource.org/license/mit
Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, this
software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
*/

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
  uint16_t addIdx[9][512];
  uint16_t rmIdx[9][512];
  int addN[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int rmN[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (const ActiveFeature& f : added) {
    if (f.group >= 9) continue;
    if (addN[f.group] < 512) addIdx[f.group][addN[f.group]++] = (uint16_t)f.index;
  }
  for (const ActiveFeature& f : removed) {
    if (f.group >= 9) continue;
    if (rmN[f.group] < 512) rmIdx[f.group][rmN[f.group]++] = (uint16_t)f.index;
  }
  for (int grp = 0; grp < 9; ++grp) {
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
