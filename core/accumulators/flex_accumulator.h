#pragma once

#include <cstdint>
#include <vector>

#include "core/accumulators/token_layout.h"
#include "core/features/feature_set.h"

namespace rune {

class FlexEmbeddings {
 public:
  FlexEmbeddings();
  void configure(int dim);
  void init(int seed);
  float get(int group, int index, int dim) const;
  void set(int group, int index, int dim, float v);
  const std::vector<float>& groupData(int group) const { return tables_[group]; }
  std::vector<float>& groupData(int group) { return tables_[group]; }
  int dim() const { return dim_; }
  size_t numFloats() const;

 private:
  int dim_ = 32;
  std::vector<float> tables_[GroupedFeatureSet::kNumGroups];
};

struct FlexScales {
  float embedding[GroupedFeatureSet::kNumGroups];
};

class FlexQuantTables {
 public:
  FlexQuantTables();
  void configure(int dim, bool useInt16);
  void quantizeFrom(const FlexEmbeddings& src, const TokenLayout& layout, FlexScales& scales);
  int get(int group, int flat) const;
  bool isInt16() const { return useInt16_; }
  int dim() const { return dim_; }

 private:
  int dim_ = 32;
  bool useInt16_ = false;
  std::vector<int> tables_[GroupedFeatureSet::kNumGroups];
};

class FlexAccumulator {
 public:
  FlexAccumulator();
  void configure(const FlexEmbeddings* tables, const TokenLayout* layout);
  void refresh(const std::vector<ActiveFeature>& features);
  void applyDiff(const std::vector<ActiveFeature>& added, const std::vector<ActiveFeature>& removed);
  void tokens(float* out) const;

 private:
  const FlexEmbeddings* tables_ = nullptr;
  const TokenLayout* layout_ = nullptr;
  std::vector<float> acc_;
};

class FlexAccumulatorInt {
 public:
  FlexAccumulatorInt();
  void bind(const FlexQuantTables* tables, const FlexScales* scales, const TokenLayout* layout);
  void refresh(const std::vector<ActiveFeature>& features);
  void applyDiff(const std::vector<ActiveFeature>& added, const std::vector<ActiveFeature>& removed);
  void tokens(float* out) const;

 private:
  const FlexQuantTables* tables_ = nullptr;
  const FlexScales* scales_ = nullptr;
  const TokenLayout* layout_ = nullptr;
  std::vector<int32_t> acc_;
};

}
