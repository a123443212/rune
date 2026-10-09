#include "core/accumulators/grouped_accumulator.h"

#include <cmath>
#include <cstdint>

namespace rune {

namespace {

uint64_t lcgNext(uint64_t& s) {
  s = s * 6364136223846793005ULL + 1442695040888963407ULL;
  return s >> 33;
}

}  // namespace

EmbeddingTables::EmbeddingTables() {
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) {
    tables_[g].assign(static_cast<size_t>(GroupedFeatureSet::vocabSize(g)) *
                          static_cast<size_t>(GroupedFeatureSet::kTokenDim),
                      0.0f);
  }
}

void EmbeddingTables::init(int seed) {
  uint64_t s = static_cast<uint64_t>(seed) * 2 + 1;
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) {
    float scale = 1.0f / static_cast<float>(GroupedFeatureSet::vocabSize(g));
    for (size_t i = 0; i < tables_[g].size(); ++i) {
      double u = static_cast<double>(lcgNext(s)) / static_cast<double>(UINT64_C(0xFFFFFFFF));
      tables_[g][i] = static_cast<float>((u - 0.5) * 2.0 * scale);
    }
  }
}

float EmbeddingTables::get(int group, int index, int dim) const {
  return tables_[group][static_cast<size_t>(index) * GroupedFeatureSet::kTokenDim + dim];
}

void EmbeddingTables::set(int group, int index, int dim, float v) {
  tables_[group][static_cast<size_t>(index) * GroupedFeatureSet::kTokenDim + dim] = v;
}

size_t EmbeddingTables::numFloats() const {
  size_t n = 0;
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) n += tables_[g].size();
  return n;
}

GroupedAccumulator::GroupedAccumulator() {
  for (int g = 0; g < kTokens; ++g)
    for (int d = 0; d < kDim; ++d) acc_[g][d] = 0.0f;
}

GroupedAccumulator::GroupedAccumulator(const EmbeddingTables* tables) : tables_(tables) {
  for (int g = 0; g < kTokens; ++g)
    for (int d = 0; d < kDim; ++d) acc_[g][d] = 0.0f;
}

void GroupedAccumulator::refresh(const std::vector<ActiveFeature>& features) {
  for (int g = 0; g < kTokens; ++g)
    for (int d = 0; d < kDim; ++d) acc_[g][d] = 0.0f;
  for (const ActiveFeature& f : features) {
    int t = GroupedFeatureSet::tokenForGroup(f.group);
    for (int d = 0; d < kDim; ++d) acc_[t][d] += tables_->get(f.group, f.index, d);
  }
}

void GroupedAccumulator::applyDiff(const std::vector<ActiveFeature>& added,
                                   const std::vector<ActiveFeature>& removed) {
  for (const ActiveFeature& f : added) {
    int t = GroupedFeatureSet::tokenForGroup(f.group);
    for (int d = 0; d < kDim; ++d) acc_[t][d] += tables_->get(f.group, f.index, d);
  }
  for (const ActiveFeature& f : removed) {
    int t = GroupedFeatureSet::tokenForGroup(f.group);
    for (int d = 0; d < kDim; ++d) acc_[t][d] -= tables_->get(f.group, f.index, d);
  }
}

void GroupedAccumulator::tokens(float* out) const {
  for (int g = 0; g < kTokens; ++g) {
    for (int d = 0; d < kDim; ++d) {
      float v = acc_[g][d];
      if (v < 0.0f) v = 0.0f;
      if (v > 1.0f) v = 1.0f;
      out[g * kDim + d] = v;
    }
  }
}

QuantEmbeddingTables::QuantEmbeddingTables() {
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) {
    tables_[g].assign(static_cast<size_t>(GroupedFeatureSet::vocabSize(g)) *
                          static_cast<size_t>(GroupedFeatureSet::kTokenDim),
                      0);
  }
}

void QuantEmbeddingTables::quantizeFrom(const EmbeddingTables& src, QuantScales& scales) {
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) {
    const std::vector<float>& data = src.groupData(g);
    float maxAbs = 0.0f;
    for (float v : data) {
      float a = v < 0 ? -v : v;
      if (a > maxAbs) maxAbs = a;
    }
    float scale = maxAbs / 127.0f;
    if (scale <= 0.0f) scale = 1.0f;
    scales.embedding[g] = scale;
    for (size_t i = 0; i < data.size(); ++i) {
      int q = static_cast<int>(std::lround(data[i] / scale));
      if (q > 127) q = 127;
      if (q < -127) q = -127;
      tables_[g][i] = static_cast<int8_t>(q);
    }
  }
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) scales.bias[g] = 0.0f;
}

void QuantEmbeddingTables::dequantizeTo(EmbeddingTables& dst, const QuantScales& scales) const {
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) {
    std::vector<float>& data = dst.groupData(g);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<float>(tables_[g][i]) * scales.embedding[g];
    }
  }
}

int8_t QuantEmbeddingTables::get(int group, int flat) const { return tables_[group][flat]; }

GroupedAccumulatorInt::GroupedAccumulatorInt() {
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g)
    for (int d = 0; d < GroupedFeatureSet::kTokenDim; ++d) acc_[g][d] = 0;
}

void GroupedAccumulatorInt::bind(const QuantEmbeddingTables* tables, const QuantScales* scales) {
  tables_ = tables;
  scales_ = scales;
}

void GroupedAccumulatorInt::refresh(const std::vector<ActiveFeature>& features) {
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g)
    for (int d = 0; d < GroupedFeatureSet::kTokenDim; ++d) acc_[g][d] = 0;
  applyDiff(features, std::vector<ActiveFeature>{});
}

void GroupedAccumulatorInt::applyDiff(const std::vector<ActiveFeature>& added,
                                      const std::vector<ActiveFeature>& removed) {
  for (const ActiveFeature& f : added) {
    size_t base = static_cast<size_t>(f.index) * GroupedFeatureSet::kTokenDim;
    for (int d = 0; d < GroupedFeatureSet::kTokenDim; ++d) {
      acc_[f.group][d] += tables_->get(f.group, static_cast<int>(base) + d);
    }
  }
  for (const ActiveFeature& f : removed) {
    size_t base = static_cast<size_t>(f.index) * GroupedFeatureSet::kTokenDim;
    for (int d = 0; d < GroupedFeatureSet::kTokenDim; ++d) {
      acc_[f.group][d] -= tables_->get(f.group, static_cast<int>(base) + d);
    }
  }
}

void GroupedAccumulatorInt::tokens(float* out) const {
  for (int t = 0; t < 8; ++t) {
    for (int d = 0; d < GroupedFeatureSet::kTokenDim; ++d) {
      out[t * GroupedFeatureSet::kTokenDim + d] = 0.0f;
    }
  }
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) {
    int t = GroupedFeatureSet::tokenForGroup(g);
    for (int d = 0; d < GroupedFeatureSet::kTokenDim; ++d) {
      out[t * GroupedFeatureSet::kTokenDim + d] +=
          static_cast<float>(acc_[g][d]) * scales_->embedding[g];
    }
  }
  for (int i = 0; i < 8 * GroupedFeatureSet::kTokenDim; ++i) {
    if (out[i] < 0.0f) out[i] = 0.0f;
    if (out[i] > 1.0f) out[i] = 1.0f;
  }
}

Quant16Tables::Quant16Tables() {
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) {
    tables_[g].assign(static_cast<size_t>(GroupedFeatureSet::vocabSize(g)) *
                          static_cast<size_t>(GroupedFeatureSet::kTokenDim),
                      0);
  }
}

void Quant16Tables::quantizeFrom(const EmbeddingTables& src, QuantScales& scales) {
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) {
    const std::vector<float>& data = src.groupData(g);
    float maxAbs = 0.0f;
    for (float v : data) {
      float a = v < 0 ? -v : v;
      if (a > maxAbs) maxAbs = a;
    }
    float scale = maxAbs / 32767.0f;
    if (scale <= 0.0f) scale = 1.0f;
    scales.embedding[g] = scale;
    for (size_t i = 0; i < data.size(); ++i) {
      int q = static_cast<int>(std::lround(data[i] / scale));
      if (q > 32767) q = 32767;
      if (q < -32767) q = -32767;
      tables_[g][i] = static_cast<int16_t>(q);
    }
  }
}

int Quant16Tables::get(int group, int flat) const { return tables_[group][flat]; }

GroupedAccumulator16::GroupedAccumulator16() {
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g)
    for (int d = 0; d < GroupedFeatureSet::kTokenDim; ++d) acc_[g][d] = 0;
}

void GroupedAccumulator16::bind(const Quant16Tables* tables, const QuantScales* scales) {
  tables_ = tables;
  scales_ = scales;
}

void GroupedAccumulator16::refresh(const std::vector<ActiveFeature>& features) {
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g)
    for (int d = 0; d < GroupedFeatureSet::kTokenDim; ++d) acc_[g][d] = 0;
  applyDiff(features, std::vector<ActiveFeature>{});
}

void GroupedAccumulator16::applyDiff(const std::vector<ActiveFeature>& added,
                                     const std::vector<ActiveFeature>& removed) {
  for (const ActiveFeature& f : added) {
    size_t base = static_cast<size_t>(f.index) * GroupedFeatureSet::kTokenDim;
    for (int d = 0; d < GroupedFeatureSet::kTokenDim; ++d) {
      acc_[f.group][d] += tables_->get(f.group, static_cast<int>(base) + d);
    }
  }
  for (const ActiveFeature& f : removed) {
    size_t base = static_cast<size_t>(f.index) * GroupedFeatureSet::kTokenDim;
    for (int d = 0; d < GroupedFeatureSet::kTokenDim; ++d) {
      acc_[f.group][d] -= tables_->get(f.group, static_cast<int>(base) + d);
    }
  }
}

void GroupedAccumulator16::tokens(float* out) const {
  for (int t = 0; t < 8; ++t) {
    for (int d = 0; d < GroupedFeatureSet::kTokenDim; ++d) {
      out[t * GroupedFeatureSet::kTokenDim + d] = 0.0f;
    }
  }
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) {
    int t = GroupedFeatureSet::tokenForGroup(g);
    for (int d = 0; d < GroupedFeatureSet::kTokenDim; ++d) {
      out[t * GroupedFeatureSet::kTokenDim + d] +=
          static_cast<float>(acc_[g][d]) * scales_->embedding[g];
    }
  }
  for (int i = 0; i < 8 * GroupedFeatureSet::kTokenDim; ++i) {
    if (out[i] < 0.0f) out[i] = 0.0f;
    if (out[i] > 1.0f) out[i] = 1.0f;
  }
}

}
