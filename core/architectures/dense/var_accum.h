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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/features/feature_set.h"

namespace rune {

struct VarWidths {
  int w[9] = {32, 32, 32, 32, 32, 32, 32, 32, 32};
  static bool make(const std::vector<int>& dims, VarWidths& out, std::string& err);
  int total() const;
};

class VarEmbeddings {
 public:
  VarEmbeddings();
  void configure(const VarWidths& widths);
  void init(int seed);
  float get(int group, int index, int dim) const;
  void set(int group, int index, int dim, float v);
  int groupWidth(int group) const { return widths_.w[group]; }
  const std::vector<float>& groupData(int group) const { return tables_[group]; }
  std::vector<float>& groupData(int group) { return tables_[group]; }
  size_t numFloats() const;

 private:
  VarWidths widths_;
  std::vector<float> tables_[GroupedFeatureSet::kNumGroups];
};

struct VarScales {
  float token[GroupedFeatureSet::kNumGroups] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
};

class VarQuantTables {
 public:
  VarQuantTables();
  void configure(const VarWidths& widths, bool useInt16);
  void quantizeFrom(const VarEmbeddings& src, VarScales& scales);
  int get(int group, int flat) const;
  bool isInt16() const { return useInt16_; }

 private:
  VarWidths widths_;
  bool useInt16_ = false;
  std::vector<int> tables_[GroupedFeatureSet::kNumGroups];
};

class VarAccumulator {
 public:
  VarAccumulator();
  void configure(const VarEmbeddings* tables);
  void refresh(const std::vector<ActiveFeature>& features);
  void applyDiff(const std::vector<ActiveFeature>& added, const std::vector<ActiveFeature>& removed);
  void tokens(float* out) const;
  int tokenOffset(int token) const;

 private:
  const VarEmbeddings* tables_ = nullptr;
  VarWidths widths_;
  std::vector<float> acc_;
};

class VarAccumulatorInt {
 public:
  VarAccumulatorInt();
  void bind(const VarQuantTables* tables, const VarScales* scales, const VarWidths& widths);
  void refresh(const std::vector<ActiveFeature>& features);
  void applyDiff(const std::vector<ActiveFeature>& added, const std::vector<ActiveFeature>& removed);
  void tokens(float* out) const;
  int tokenOffset(int token) const;

 private:
  const VarQuantTables* tables_ = nullptr;
  const VarScales* scales_ = nullptr;
  VarWidths widths_;
  std::vector<int32_t> acc_;
};

}
