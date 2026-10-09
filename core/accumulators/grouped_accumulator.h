#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "core/features/feature_set.h"

namespace rune {

class EmbeddingTables {
 public:
  EmbeddingTables();
  explicit EmbeddingTables(const int* vocabs);
  void init(int seed);
  float get(int group, int index, int dim) const;
  void set(int group, int index, int dim, float v);
  std::vector<float>& groupData(int group) { return tables_[group]; }
  const std::vector<float>& groupData(int group) const { return tables_[group]; }
  size_t numFloats() const;

 private:
  std::vector<float> tables_[GroupedFeatureSet::kNumGroups];
};

class GroupedAccumulator {
 public:
  GroupedAccumulator();
  explicit GroupedAccumulator(const EmbeddingTables* tables);

  void bind(const EmbeddingTables* tables) { tables_ = tables; }
  void refresh(const std::vector<ActiveFeature>& features);
  void applyDiff(const std::vector<ActiveFeature>& added, const std::vector<ActiveFeature>& removed);
  void tokens(float* out) const;

  static constexpr int kTokens = 8;
  static constexpr int kDim = GroupedFeatureSet::kTokenDim;

 private:
  const EmbeddingTables* tables_ = nullptr;
  alignas(64) float acc_[8][GroupedFeatureSet::kTokenDim];
};

struct QuantScales {
  float embedding[GroupedFeatureSet::kNumGroups];
  float bias[GroupedFeatureSet::kNumGroups];
};

class QuantEmbeddingTables {
 public:
  QuantEmbeddingTables();
  void quantizeFrom(const EmbeddingTables& src, QuantScales& scales);
  void dequantizeTo(EmbeddingTables& dst, const QuantScales& scales) const;
  int8_t get(int group, int flat) const;
  const std::vector<int8_t>& groupData(int group) const { return tables_[group]; }

 private:
  std::vector<int8_t> tables_[GroupedFeatureSet::kNumGroups];
};

class GroupedAccumulatorInt {
 public:
  GroupedAccumulatorInt();
  void bind(const QuantEmbeddingTables* tables, const QuantScales* scales);
  void refresh(const std::vector<ActiveFeature>& features);
  void applyDiff(const std::vector<ActiveFeature>& added, const std::vector<ActiveFeature>& removed);
  void tokens(float* out) const;

 private:
  const QuantEmbeddingTables* tables_ = nullptr;
  const QuantScales* scales_ = nullptr;
  alignas(64) int32_t acc_[GroupedFeatureSet::kNumGroups][GroupedFeatureSet::kTokenDim];
};

class Quant16Tables {
 public:
  Quant16Tables();
  void quantizeFrom(const EmbeddingTables& src, QuantScales& scales);
  int get(int group, int flat) const;
  const std::vector<int16_t>& groupData(int group) const { return tables_[group]; }

 private:
  std::vector<int16_t> tables_[GroupedFeatureSet::kNumGroups];
};

class GroupedAccumulator16 {
 public:
  GroupedAccumulator16();
  void bind(const Quant16Tables* tables, const QuantScales* scales);
  void refresh(const std::vector<ActiveFeature>& features);
  void applyDiff(const std::vector<ActiveFeature>& added, const std::vector<ActiveFeature>& removed);
  void tokens(float* out) const;

 private:
  const Quant16Tables* tables_ = nullptr;
  const QuantScales* scales_ = nullptr;
  alignas(64) int32_t acc_[GroupedFeatureSet::kNumGroups][GroupedFeatureSet::kTokenDim];
};

}
