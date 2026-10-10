#include "core/architectures/sfnn/sfnn.h"

#include <cmath>
#include <string>

#include "core/kernels/fused.h"
#include "core/simd/simd.h"

namespace rune {

namespace {

void initVec(std::vector<float>& values, size_t count, uint64_t& state, float scale) {
  values.assign(count, 0.0f);
  for (size_t i = 0; i < count; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    double unit = static_cast<double>(state >> 33) / static_cast<double>(0xFFFFFFFFULL);
    values[i] = static_cast<float>((unit - 0.5) * 2.0 * scale);
  }
}

}

SfnnBaseline::SfnnBaseline() {
  uint64_t state = 777;
  heads_.resize(1);
  initVec(heads_[0].w1, kH1 * kIn, state, 0.04f);
  initVec(heads_[0].b1, kH1, state, 0.01f);
  initVec(heads_[0].w2, kH2 * kH1, state, 0.05f);
  initVec(heads_[0].b2, kH2, state, 0.01f);
  initVec(heads_[0].wvo, kH2, state, 0.05f);
  initVec(heads_[0].bvo, 1, state, 0.01f);
  initVec(heads_[0].wwdl, 3 * kH2, state, 0.05f);
  initVec(heads_[0].bwdl, 3, state, 0.01f);
  scratch_.assign(kH1 + 2 * kH1 + kH2, 0.0f);
}

const HeadBucket& SfnnBaseline::headFor(int phase) const {
  if (heads_.size() > 1) {
    int bucket = phase;
    if (bucket < 0) bucket = 0;
    if (bucket > 2) bucket = 2;
    return heads_[static_cast<size_t>(bucket)];
  }
  return heads_[0];
}

void SfnnBaseline::forward(const float* tokens, float& value, float* wdl, int phase) const {
  const HeadBucket& head = headFor(phase);
  int hidden1 = static_cast<int>(head.b1.size());
  int hidden2 = static_cast<int>(head.b2.size());
  bool pair = head.w2.size() == static_cast<size_t>(hidden2) * static_cast<size_t>(hidden1) * 2;
  if (pair) {
    float* pre = scratch_.data();
    float* paired = scratch_.data() + hidden1;
    float* hidden = scratch_.data() + hidden1 + hidden1 * 2;
    simd::matVec(head.w1.data(), tokens, head.b1.data(), pre, hidden1, kIn);
    for (int i = 0; i < hidden1; ++i) {
      float clipped = pre[i] < 0.0f ? 0.0f : (pre[i] > 1.0f ? 1.0f : pre[i]);
      paired[i] = clipped;
      paired[hidden1 + i] = clipped * clipped;
    }
    simd::matVecClipped(head.w2.data(), paired, head.b2.data(), hidden, hidden2, hidden1 * 2);
    float result = head.bvo[0];
    for (int i = 0; i < hidden2; ++i) result += head.wvo[i] * hidden[i];
    value = std::tanh(result);
    simd::matVec(head.wwdl.data(), hidden, head.bwdl.data(), wdl, 3, hidden2);
    return;
  }
  float* hidden1Values = scratch_.data();
  float* hidden2Values = scratch_.data() + kH1;
  simd::matVecClipped(head.w1.data(), tokens, head.b1.data(), hidden1Values, kH1, kIn);
  simd::matVecClipped(head.w2.data(), hidden1Values, head.b2.data(), hidden2Values, kH2, kH1);
  float result = head.bvo[0];
  for (int i = 0; i < kH2; ++i) result += head.wvo[i] * hidden2Values[i];
  value = std::tanh(result);
  simd::matVec(head.wwdl.data(), hidden2Values, head.bwdl.data(), wdl, 3, kH2);
}

size_t SfnnBaseline::parameterCount() const {
  size_t count = 0;
  for (const HeadBucket& head : heads_) {
    count += head.w1.size() + head.b1.size() + head.w2.size() + head.b2.size() + head.wvo.size() +
             head.bvo.size() + head.wwdl.size() + head.bwdl.size();
  }
  return count;
}

size_t SfnnBaseline::modelSizeBytes() const { return parameterCount() * 4; }

void SfnnBaseline::getTensors(std::vector<std::string>& names, std::vector<std::vector<int>>& shapes,
                              std::vector<const float*>& data) const {
  names.clear();
  shapes.clear();
  data.clear();
  for (size_t bucket = 0; bucket < heads_.size(); ++bucket) {
    std::string suffix = heads_.size() > 1 ? "_b" + std::to_string(bucket) : "";
    const HeadBucket& head = heads_[bucket];
    names.insert(names.end(), {"w1" + suffix, "b1" + suffix, "w2" + suffix, "b2" + suffix,
                               "wv" + suffix, "bv" + suffix, "wwdl" + suffix, "bwdl" + suffix});
    int hidden1 = static_cast<int>(head.b1.size());
    int hidden2 = static_cast<int>(head.b2.size());
    int w2Columns = hidden2 == 0 ? 0 : static_cast<int>(head.w2.size() / static_cast<size_t>(hidden2));
    shapes.insert(shapes.end(), {{hidden1, kIn}, {hidden1}, {hidden2, w2Columns}, {hidden2},
                                 {1, hidden2}, {1}, {3, hidden2}, {3}});
    data.insert(data.end(), {head.w1.data(), head.b1.data(), head.w2.data(), head.b2.data(),
                             head.wvo.data(), head.bvo.data(), head.wwdl.data(), head.bwdl.data()});
  }
}

bool SfnnBaseline::setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) {
  std::vector<std::string> expected = {"w1", "b1", "w2", "b2", "wv", "bv", "wwdl", "bwdl"};
  size_t buckets = 1;
  if (names.size() > expected.size()) {
    buckets = 3;
    expected.clear();
    for (int bucket = 0; bucket < 3; ++bucket) {
      std::string suffix = "_b" + std::to_string(bucket);
      expected.insert(expected.end(), {"w1" + suffix, "b1" + suffix, "w2" + suffix,
                                       "b2" + suffix, "wv" + suffix, "bv" + suffix,
                                       "wwdl" + suffix, "bwdl" + suffix});
    }
  }
  if (names != expected) return false;
  size_t singleSize = kH1 * kIn + kH1 + kH2 * kH1 + kH2 + kH2 + 1 + 3 * kH2 + 3;
  size_t pairSize = kH1 * kIn + kH1 + kH2 * kH1 * 2 + kH2 + kH2 + 1 + 3 * kH2 + 3;
  bool pair = flat.size() == pairSize * buckets;
  if (!pair && flat.size() != singleSize * buckets) return false;
  size_t w2Size = pair ? kH2 * kH1 * 2 : kH2 * kH1;
  heads_.clear();
  size_t offset = 0;
  for (size_t bucket = 0; bucket < buckets; ++bucket) {
    HeadBucket head;
    head.w1.assign(kH1 * kIn, 0.0f);
    head.b1.assign(kH1, 0.0f);
    head.w2.assign(w2Size, 0.0f);
    head.b2.assign(kH2, 0.0f);
    head.wvo.assign(kH2, 0.0f);
    head.bvo.assign(1, 0.0f);
    head.wwdl.assign(3 * kH2, 0.0f);
    head.bwdl.assign(3, 0.0f);
    std::vector<std::vector<float>*> slots = {&head.w1, &head.b1, &head.w2, &head.b2,
                                              &head.wvo, &head.bvo, &head.wwdl, &head.bwdl};
    for (auto* slot : slots) {
      if (offset + slot->size() > flat.size()) return false;
      for (size_t i = 0; i < slot->size(); ++i) (*slot)[i] = flat[offset + i];
      offset += slot->size();
    }
    heads_.push_back(std::move(head));
  }
  return offset == flat.size();
}

ModelSpec SfnnBaseline::spec() const {
  ModelSpec modelSpec;
  modelSpec.arch = archId();
  bool pair = !heads_.empty() &&
              heads_[0].w2.size() == heads_[0].b2.size() * heads_[0].b1.size() * 2;
  modelSpec.archVersion = pair ? "0.2.0" : archVersion();
  modelSpec.attention = "none";
  modelSpec.geometricBias = "none";
  modelSpec.headBuckets = static_cast<int>(heads_.size());
  return modelSpec;
}

}