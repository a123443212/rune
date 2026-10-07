#include "core/accumulators/flex_accumulator.h"

#include <cmath>

namespace rune {

namespace {

uint64_t lcgNext(uint64_t& s) {
  s = s * 6364136223846793005ULL + 1442695040888963407ULL;
  return s >> 33;
}

}  // namespace

FlexEmbeddings::FlexEmbeddings() { configure(32); }

void FlexEmbeddings::configure(int dim) {
  dim_ = dim;
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) {
    tables_[g].assign(static_cast<size_t>(GroupedFeatureSet::vocabSize(g)) * static_cast<size_t>(dim),
                      0.0f);
  }
}

void FlexEmbeddings::init(int seed) {
  uint64_t s = static_cast<uint64_t>(seed) * 2 + 1;
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) {
    float scale = 1.0f / static_cast<float>(GroupedFeatureSet::vocabSize(g));
    for (size_t i = 0; i < tables_[g].size(); ++i) {
      double u = static_cast<double>(lcgNext(s)) / static_cast<double>(UINT64_C(0xFFFFFFFF));
      tables_[g][i] = static_cast<float>((u - 0.5) * 2.0 * scale);
    }
  }
}

float FlexEmbeddings::get(int group, int index, int dim) const {
  return tables_[group][static_cast<size_t>(index) * dim_ + dim];
}

void FlexEmbeddings::set(int group, int index, int dim, float v) {
  tables_[group][static_cast<size_t>(index) * dim_ + dim] = v;
}

size_t FlexEmbeddings::numFloats() const {
  size_t n = 0;
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) n += tables_[g].size();
  return n;
}

FlexQuantTables::FlexQuantTables() { configure(32, false); }

void FlexQuantTables::configure(int dim, bool useInt16) {
  dim_ = dim;
  useInt16_ = useInt16;
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) {
    tables_[g].assign(static_cast<size_t>(GroupedFeatureSet::vocabSize(g)) * static_cast<size_t>(dim),
                      0);
  }
}

void FlexQuantTables::quantizeFrom(const FlexEmbeddings& src, const TokenLayout& layout,
                                   FlexScales& scales) {
  float bound = useInt16_ ? 32767.0f : 127.0f;
  int qmax = useInt16_ ? 32767 : 127;
  int qmin = useInt16_ ? -32767 : -127;
  (void)layout;
  for (int g = 0; g < GroupedFeatureSet::kNumGroups; ++g) {
    const std::vector<float>& data = src.groupData(g);
    float maxAbs = 0.0f;
    for (float v : data) {
      float a = v < 0 ? -v : v;
      if (a > maxAbs) maxAbs = a;
    }
    float scale = maxAbs / bound;
    if (scale <= 0.0f) scale = 1.0f;
    scales.embedding[g] = scale;
    for (size_t i = 0; i < data.size(); ++i) {
      int q = static_cast<int>(std::lround(data[i] / scale));
      if (q > qmax) q = qmax;
      if (q < qmin) q = qmin;
      tables_[g][i] = q;
    }
  }
}

int FlexQuantTables::get(int group, int flat) const { return tables_[group][flat]; }

FlexAccumulator::FlexAccumulator() {
  TokenLayout fallback;
  std::string err;
  TokenLayout::make(8, 32, fallback, err);
  static TokenLayout kept = fallback;
  layout_ = &kept;
}

void FlexAccumulator::configure(const FlexEmbeddings* tables, const TokenLayout* layout) {
  tables_ = tables;
  layout_ = layout;
  acc_.assign(static_cast<size_t>(layout->tokens) * static_cast<size_t>(layout->dim), 0.0f);
}

void FlexAccumulator::refresh(const std::vector<ActiveFeature>& features) {
  std::fill(acc_.begin(), acc_.end(), 0.0f);
  applyDiff(features, std::vector<ActiveFeature>{});
}

void FlexAccumulator::applyDiff(const std::vector<ActiveFeature>& added,
                                const std::vector<ActiveFeature>& removed) {
  int d = layout_->dim;
  for (const ActiveFeature& f : added) {
    int t = layout_->findToken(f.group, f.index);
    if (t < 0) continue;
    for (int k = 0; k < d; ++k) acc_[t * d + k] += tables_->get(f.group, f.index, k);
  }
  for (const ActiveFeature& f : removed) {
    int t = layout_->findToken(f.group, f.index);
    if (t < 0) continue;
    for (int k = 0; k < d; ++k) acc_[t * d + k] -= tables_->get(f.group, f.index, k);
  }
}

void FlexAccumulator::tokens(float* out) const {
  for (size_t i = 0; i < acc_.size(); ++i) {
    float v = acc_[i];
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    out[i] = v;
  }
}

FlexAccumulatorInt::FlexAccumulatorInt() {}

void FlexAccumulatorInt::bind(const FlexQuantTables* tables, const FlexScales* scales,
                              const TokenLayout* layout) {
  tables_ = tables;
  scales_ = scales;
  layout_ = layout;
  acc_.assign(static_cast<size_t>(layout->tokens) * static_cast<size_t>(layout->dim), 0);
}

void FlexAccumulatorInt::refresh(const std::vector<ActiveFeature>& features) {
  std::fill(acc_.begin(), acc_.end(), 0);
  applyDiff(features, std::vector<ActiveFeature>{});
}

void FlexAccumulatorInt::applyDiff(const std::vector<ActiveFeature>& added,
                                   const std::vector<ActiveFeature>& removed) {
  int d = layout_->dim;
  for (const ActiveFeature& f : added) {
    int t = layout_->findToken(f.group, f.index);
    if (t < 0) continue;
    size_t base = static_cast<size_t>(f.index) * d;
    for (int k = 0; k < d; ++k) acc_[t * d + k] += tables_->get(f.group, static_cast<int>(base) + k);
  }
  for (const ActiveFeature& f : removed) {
    int t = layout_->findToken(f.group, f.index);
    if (t < 0) continue;
    size_t base = static_cast<size_t>(f.index) * d;
    for (int k = 0; k < d; ++k) acc_[t * d + k] -= tables_->get(f.group, static_cast<int>(base) + k);
  }
}

void FlexAccumulatorInt::tokens(float* out) const {
  int n = layout_->tokens;
  int d = layout_->dim;
  for (int t = 0; t < n; ++t) {
    int g = layout_->sources[t][0].group;
    float scale = scales_->embedding[g];
    for (int k = 0; k < d; ++k) {
      float v = static_cast<float>(acc_[t * d + k]) * scale;
      if (v < 0.0f) v = 0.0f;
      if (v > 1.0f) v = 1.0f;
      out[t * d + k] = v;
    }
  }
}

}
