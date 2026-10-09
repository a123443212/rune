#include "core/architectures/mlp/mlp.h"

#include <cmath>
#include <string>
#include "core/kernels/fused.h"

namespace rune {

namespace {

void initVec(std::vector<float>& v, size_t n, uint64_t& s, float scale) {
  v.assign(n, 0.0f);
  for (size_t i = 0; i < n; ++i) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    double u = static_cast<double>(s >> 33) / static_cast<double>(0xFFFFFFFFULL);
    v[i] = static_cast<float>((u - 0.5) * 2.0 * scale);
  }
}

}  // namespace

GroupedMlp::GroupedMlp() {
  uint64_t s = 12345;
  heads_.resize(1);
  initVec(heads_[0].w1, 128 * 256, s, 0.05f);
  initVec(heads_[0].b1, 128, s, 0.01f);
  initVec(heads_[0].w2, 32 * 128, s, 0.05f);
  initVec(heads_[0].b2, 32, s, 0.01f);
  initVec(heads_[0].wvo, 1 * 32, s, 0.05f);
  initVec(heads_[0].bvo, 1, s, 0.01f);
  initVec(heads_[0].wwdl, 3 * 32, s, 0.05f);
  initVec(heads_[0].bwdl, 3, s, 0.01f);
  scratch_.assign(160, 0.0f);
}

const HeadBucket& GroupedMlp::headFor(int phase) const {
  if (heads_.size() > 1) {
    int b = phase;
    if (b < 0) b = 0;
    if (b > 2) b = 2;
    return heads_[static_cast<size_t>(b)];
  }
  return heads_[0];
}

void GroupedMlp::forward(const float* tokens, float& value, float* wdl, int phase) const {
  const HeadBucket& h = headFor(phase);
  float* h1 = scratch_.data();
  float* h2 = scratch_.data() + 128;
  simd::matVecClipped(h.w1.data(), tokens, h.b1.data(), h1, kH1, kIn);
  simd::matVecClipped(h.w2.data(), h1, h.b2.data(), h2, kH2, kH1);
  float v = h.bvo[0];
  for (int i = 0; i < kH2; ++i) v += h.wvo[i] * h2[i];
  value = std::tanh(v);
  simd::matVec(h.wwdl.data(), h2, h.bwdl.data(), wdl, 3, kH2);
}

size_t GroupedMlp::parameterCount() const {
  size_t n = 0;
  for (const HeadBucket& h : heads_) {
    n += h.w1.size() + h.b1.size() + h.w2.size() + h.b2.size() + h.wvo.size() +
         h.bvo.size() + h.wwdl.size() + h.bwdl.size();
  }
  return n;
}

size_t GroupedMlp::modelSizeBytes() const { return parameterCount() * 4; }
void GroupedMlp::getTensors(std::vector<std::string>& names,
                               std::vector<std::vector<int>>& shapes,
                               std::vector<const float*>& data) const {
  names.clear();
  shapes.clear();
  data.clear();
  for (size_t b = 0; b < heads_.size(); ++b) {
    std::string suffix = heads_.size() > 1 ? "_b" + std::to_string(b) : "";
    const HeadBucket& h = heads_[b];
    names.insert(names.end(), {"w1" + suffix, "b1" + suffix, "w2" + suffix, "b2" + suffix,
                               "wv" + suffix, "bv" + suffix, "wwdl" + suffix, "bwdl" + suffix});
    shapes.insert(shapes.end(), {{128, 256}, {128}, {32, 128}, {32}, {1, 32}, {1}, {3, 32}, {3}});
    data.insert(data.end(), {h.w1.data(), h.b1.data(), h.w2.data(), h.b2.data(), h.wvo.data(),
                             h.bvo.data(), h.wwdl.data(), h.bwdl.data()});
  }
}

bool GroupedMlp::setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) {
  std::vector<std::string> base = {"w1", "b1", "w2", "b2", "wv", "bv", "wwdl", "bwdl"};
  size_t nbuckets = 1;
  if (names.size() > base.size()) {
    nbuckets = 3;
    base.clear();
    for (int b = 0; b < 3; ++b) {
      std::string suffix = "_b" + std::to_string(b);
      base.insert(base.end(), {"w1" + suffix, "b1" + suffix, "w2" + suffix, "b2" + suffix,
                               "wv" + suffix, "bv" + suffix, "wwdl" + suffix, "bwdl" + suffix});
    }
  }
  if (names.size() != base.size()) return false;
  for (size_t i = 0; i < base.size(); ++i) {
    if (names[i] != base[i]) return false;
  }
  heads_.clear();
  size_t off = 0;
  for (size_t b = 0; b < nbuckets; ++b) {
    HeadBucket h;
    h.w1.assign(128 * 256, 0.0f);
    h.b1.assign(128, 0.0f);
    h.w2.assign(32 * 128, 0.0f);
    h.b2.assign(32, 0.0f);
    h.wvo.assign(32, 0.0f);
    h.bvo.assign(1, 0.0f);
    h.wwdl.assign(3 * 32, 0.0f);
    h.bwdl.assign(3, 0.0f);
    std::vector<std::vector<float>*> slots = {&h.w1, &h.b1, &h.w2, &h.b2,
                                              &h.wvo, &h.bvo, &h.wwdl, &h.bwdl};
    for (auto* slot : slots) {
      if (off + slot->size() > flat.size()) return false;
      for (size_t k = 0; k < slot->size(); ++k) (*slot)[k] = flat[off + k];
      off += slot->size();
    }
    heads_.push_back(std::move(h));
  }
  return off == flat.size();
}

ModelSpec GroupedMlp::spec() const {
  ModelSpec s;
  s.arch = archId();
  s.archVersion = archVersion();
  s.attention = "none";
  s.geometricBias = "none";
  s.headBuckets = static_cast<int>(heads_.size());
  return s;
}

SfnnBaseline::SfnnBaseline() {
  uint64_t s = 777;
  heads_.resize(1);
  initVec(heads_[0].w1, 256 * 256, s, 0.04f);
  initVec(heads_[0].b1, 256, s, 0.01f);
  initVec(heads_[0].w2, 32 * 256, s, 0.05f);
  initVec(heads_[0].b2, 32, s, 0.01f);
  initVec(heads_[0].wvo, 1 * 32, s, 0.05f);
  initVec(heads_[0].bvo, 1, s, 0.01f);
  initVec(heads_[0].wwdl, 3 * 32, s, 0.05f);
  initVec(heads_[0].bwdl, 3, s, 0.01f);
  scratch_.assign(288, 0.0f);
}

const HeadBucket& SfnnBaseline::headFor(int phase) const {
  if (heads_.size() > 1) {
    int b = phase;
    if (b < 0) b = 0;
    if (b > 2) b = 2;
    return heads_[static_cast<size_t>(b)];
  }
  return heads_[0];
}

void SfnnBaseline::forward(const float* tokens, float& value, float* wdl, int phase) const {
  const HeadBucket& h = headFor(phase);
  float* h1 = scratch_.data();
  float* h2 = scratch_.data() + 256;
  simd::matVecClipped(h.w1.data(), tokens, h.b1.data(), h1, kH1, kIn);
  simd::matVecClipped(h.w2.data(), h1, h.b2.data(), h2, kH2, kH1);
  float v = h.bvo[0];
  for (int i = 0; i < kH2; ++i) v += h.wvo[i] * h2[i];
  value = std::tanh(v);
  simd::matVec(h.wwdl.data(), h2, h.bwdl.data(), wdl, 3, kH2);
}

size_t SfnnBaseline::parameterCount() const {
  size_t n = 0;
  for (const HeadBucket& h : heads_) {
    n += h.w1.size() + h.b1.size() + h.w2.size() + h.b2.size() + h.wvo.size() +
         h.bvo.size() + h.wwdl.size() + h.bwdl.size();
  }
  return n;
}

size_t SfnnBaseline::modelSizeBytes() const { return parameterCount() * 4; }
void SfnnBaseline::getTensors(std::vector<std::string>& names,
                                 std::vector<std::vector<int>>& shapes,
                                 std::vector<const float*>& data) const {
  names.clear();
  shapes.clear();
  data.clear();
  for (size_t b = 0; b < heads_.size(); ++b) {
    std::string suffix = heads_.size() > 1 ? "_b" + std::to_string(b) : "";
    const HeadBucket& h = heads_[b];
    names.insert(names.end(), {"w1" + suffix, "b1" + suffix, "w2" + suffix, "b2" + suffix,
                               "wv" + suffix, "bv" + suffix, "wwdl" + suffix, "bwdl" + suffix});
    shapes.insert(shapes.end(), {{256, 256}, {256}, {32, 256}, {32}, {1, 32}, {1}, {3, 32}, {3}});
    data.insert(data.end(), {h.w1.data(), h.b1.data(), h.w2.data(), h.b2.data(), h.wvo.data(),
                             h.bvo.data(), h.wwdl.data(), h.bwdl.data()});
  }
}

bool SfnnBaseline::setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) {
  std::vector<std::string> base = {"w1", "b1", "w2", "b2", "wv", "bv", "wwdl", "bwdl"};
  size_t nbuckets = 1;
  if (names.size() > base.size()) {
    nbuckets = 3;
    base.clear();
    for (int b = 0; b < 3; ++b) {
      std::string suffix = "_b" + std::to_string(b);
      base.insert(base.end(), {"w1" + suffix, "b1" + suffix, "w2" + suffix, "b2" + suffix,
                               "wv" + suffix, "bv" + suffix, "wwdl" + suffix, "bwdl" + suffix});
    }
  }
  if (names.size() != base.size()) return false;
  for (size_t i = 0; i < base.size(); ++i) {
    if (names[i] != base[i]) return false;
  }
  heads_.clear();
  size_t off = 0;
  for (size_t b = 0; b < nbuckets; ++b) {
    HeadBucket h;
    h.w1.assign(256 * 256, 0.0f);
    h.b1.assign(256, 0.0f);
    h.w2.assign(32 * 256, 0.0f);
    h.b2.assign(32, 0.0f);
    h.wvo.assign(32, 0.0f);
    h.bvo.assign(1, 0.0f);
    h.wwdl.assign(3 * 32, 0.0f);
    h.bwdl.assign(3, 0.0f);
    std::vector<std::vector<float>*> slots = {&h.w1, &h.b1, &h.w2, &h.b2,
                                              &h.wvo, &h.bvo, &h.wwdl, &h.bwdl};
    for (auto* slot : slots) {
      if (off + slot->size() > flat.size()) return false;
      for (size_t k = 0; k < slot->size(); ++k) (*slot)[k] = flat[off + k];
      off += slot->size();
    }
    heads_.push_back(std::move(h));
  }
  return off == flat.size();
}

ModelSpec SfnnBaseline::spec() const {
  ModelSpec s;
  s.arch = archId();
  s.archVersion = archVersion();
  s.attention = "none";
  s.geometricBias = "none";
  s.headBuckets = static_cast<int>(heads_.size());
  return s;
}

}
