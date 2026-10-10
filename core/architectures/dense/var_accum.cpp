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

#include "core/architectures/dense/var_accum.h"

#include <cmath>

namespace rune {

namespace {

uint64_t lcgNext(uint64_t& s) {
  s = s * 6364136223846793005ULL + 1442695040888963407ULL;
  return s >> 33;
}

}  // namespace

bool VarWidths::make(const std::vector<int>& dims, VarWidths& out, std::string& err) {
  if (dims.size() != 8) {
    err = "token dims must have 8 entries";
    return false;
  }
  for (int d : dims) {
    if (d < 8 || d > 64) {
      err = "token dim out of range";
      return false;
    }
  }
  for (int i = 0; i < 8; ++i) out.w[i] = dims[i];
  out.w[8] = out.w[0];
  return true;
}

int VarWidths::total() const {
  int n = 0;
  for (int i = 0; i < 8; ++i) n += w[i];
  return n;
}

VarEmbeddings::VarEmbeddings() {
  VarWidths dflt;
  configure(dflt);
}

void VarEmbeddings::configure(const VarWidths& widths) {
  widths_ = widths;
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) {
    tables_[g].assign(static_cast<size_t>(GroupedFeatureSet::vocabSize(g)) *
                          static_cast<size_t>(widths_.w[g]),
                      0.0f);
  }
}

void VarEmbeddings::init(int seed) {
  uint64_t s = static_cast<uint64_t>(seed) * 2 + 1;
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) {
    float scale = 1.0f / static_cast<float>(GroupedFeatureSet::vocabSize(g));
    for (size_t i = 0; i < tables_[g].size(); ++i) {
      double u = static_cast<double>(lcgNext(s)) / static_cast<double>(UINT64_C(0xFFFFFFFF));
      tables_[g][i] = static_cast<float>((u - 0.5) * 2.0 * scale);
    }
  }
}

float VarEmbeddings::get(int group, int index, int dim) const {
  return tables_[group][static_cast<size_t>(index) * widths_.w[group] + dim];
}

void VarEmbeddings::set(int group, int index, int dim, float v) {
  tables_[group][static_cast<size_t>(index) * widths_.w[group] + dim] = v;
}

size_t VarEmbeddings::numFloats() const {
  size_t n = 0;
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) n += tables_[g].size();
  return n;
}

VarQuantTables::VarQuantTables() {
  VarWidths dflt;
  configure(dflt, false);
}

void VarQuantTables::configure(const VarWidths& widths, bool useInt16) {
  widths_ = widths;
  useInt16_ = useInt16;
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) {
    tables_[g].assign(static_cast<size_t>(GroupedFeatureSet::vocabSize(g)) *
                          static_cast<size_t>(widths_.w[g]),
                      0);
  }
}

void VarQuantTables::quantizeFrom(const VarEmbeddings& src, VarScales& scales) {
  float bound = useInt16_ ? 32767.0f : 127.0f;
  int qmax = useInt16_ ? 32767 : 127;
  int qmin = useInt16_ ? -32767 : -127;
  for (int t = 0; t < 8; ++t) {
    const std::vector<float>& data = src.groupData(t);
    float maxAbs = 0.0f;
    for (float v : data) {
      float a = v < 0 ? -v : v;
      if (a > maxAbs) maxAbs = a;
    }
    const std::vector<float>* extra = nullptr;
    if (t == 0) {
      extra = &src.groupData(8);
      for (float v : *extra) {
        float a = v < 0 ? -v : v;
        if (a > maxAbs) maxAbs = a;
      }
    }
    float scale = maxAbs / bound;
    if (scale <= 0.0f) scale = 1.0f;
    scales.token[t] = scale;
    for (size_t i = 0; i < data.size(); ++i) {
      int q = static_cast<int>(std::lround(data[i] / scale));
      if (q > qmax) q = qmax;
      if (q < qmin) q = qmin;
      tables_[t][i] = q;
    }
    if (extra) {
      for (size_t i = 0; i < extra->size(); ++i) {
        int q = static_cast<int>(std::lround((*extra)[i] / scale));
        if (q > qmax) q = qmax;
        if (q < qmin) q = qmin;
        tables_[8][i] = q;
      }
    }
  }
  scales.token[8] = scales.token[0];
}

int VarQuantTables::get(int group, int flat) const { return tables_[group][flat]; }

VarAccumulator::VarAccumulator() {}

void VarAccumulator::configure(const VarEmbeddings* tables) {
  tables_ = tables;
  for (int g = 0; g < 9; ++g) widths_.w[g] = tables->groupWidth(g);
  int n = widths_.total();
  acc_.assign(n, 0.0f);
}

void VarAccumulator::refresh(const std::vector<ActiveFeature>& features) {
  std::fill(acc_.begin(), acc_.end(), 0.0f);
  applyDiff(features, std::vector<ActiveFeature>{});
}

void VarAccumulator::applyDiff(const std::vector<ActiveFeature>& added,
                               const std::vector<ActiveFeature>& removed) {
  for (const ActiveFeature& f : added) {
    int off = tokenOffset(GroupedFeatureSet::tokenForGroup(f.group));
    int w = widths_.w[f.group];
    for (int d = 0; d < w; ++d) acc_[off + d] += tables_->get(f.group, f.index, d);
  }
  for (const ActiveFeature& f : removed) {
    int off = tokenOffset(GroupedFeatureSet::tokenForGroup(f.group));
    int w = widths_.w[f.group];
    for (int d = 0; d < w; ++d) acc_[off + d] -= tables_->get(f.group, f.index, d);
  }
}

void VarAccumulator::tokens(float* out) const {
  for (size_t i = 0; i < acc_.size(); ++i) {
    float v = acc_[i];
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    out[i] = v;
  }
}

int VarAccumulator::tokenOffset(int token) const {
  int off = 0;
  for (int i = 0; i < token; ++i) off += widths_.w[i];
  return off;
}

VarAccumulatorInt::VarAccumulatorInt() {}

void VarAccumulatorInt::bind(const VarQuantTables* tables, const VarScales* scales,
                             const VarWidths& widths) {
  tables_ = tables;
  scales_ = scales;
  widths_ = widths;
  acc_.assign(widths_.total(), 0);
}

void VarAccumulatorInt::refresh(const std::vector<ActiveFeature>& features) {
  std::fill(acc_.begin(), acc_.end(), 0);
  applyDiff(features, std::vector<ActiveFeature>{});
}

void VarAccumulatorInt::applyDiff(const std::vector<ActiveFeature>& added,
                                  const std::vector<ActiveFeature>& removed) {
  for (const ActiveFeature& f : added) {
    int off = tokenOffset(GroupedFeatureSet::tokenForGroup(f.group));
    int w = widths_.w[f.group];
    size_t base = static_cast<size_t>(f.index) * w;
    for (int d = 0; d < w; ++d) acc_[off + d] += tables_->get(f.group, static_cast<int>(base) + d);
  }
  for (const ActiveFeature& f : removed) {
    int off = tokenOffset(GroupedFeatureSet::tokenForGroup(f.group));
    int w = widths_.w[f.group];
    size_t base = static_cast<size_t>(f.index) * w;
    for (int d = 0; d < w; ++d) acc_[off + d] -= tables_->get(f.group, static_cast<int>(base) + d);
  }
}

void VarAccumulatorInt::tokens(float* out) const {
  for (int t = 0; t < 8; ++t) {
    int off = tokenOffset(t);
    for (int d = 0; d < widths_.w[t]; ++d) {
      float v = static_cast<float>(acc_[off + d]) * scales_->token[t];
      if (v < 0.0f) v = 0.0f;
      if (v > 1.0f) v = 1.0f;
      out[off + d] = v;
    }
  }
}

int VarAccumulatorInt::tokenOffset(int token) const {
  int off = 0;
  for (int i = 0; i < token; ++i) off += widths_.w[i];
  return off;
}

}
