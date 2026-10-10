#include "core/architectures/mlp/mlp_small.h"
#include <cmath>
#include "core/simd/simd.h"
namespace rune {
namespace {
void initSmall(std::vector<float>& v, size_t n, uint64_t& s, float scale) {
  v.assign(n, 0.0f);
  for (size_t i = 0; i < n; ++i) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    double u = static_cast<double>(s >> 33) / static_cast<double>(0xFFFFFFFFULL);
    v[i] = static_cast<float>((u - 0.5) * 2.0 * scale);
  }
}
}
GroupedMlpSmall::GroupedMlpSmall() {
  uint64_t s = 9101;
  heads_.resize(1);
  initSmall(heads_[0].w1, kH1 * kIn, s, 0.05f);
  initSmall(heads_[0].b1, kH1, s, 0.01f);
  initSmall(heads_[0].w2, kH2 * kH1, s, 0.05f);
  initSmall(heads_[0].b2, kH2, s, 0.01f);
  initSmall(heads_[0].wvo, 1 * kH2, s, 0.05f);
  initSmall(heads_[0].bvo, 1, s, 0.01f);
  initSmall(heads_[0].wwdl, 3 * kH2, s, 0.05f);
  initSmall(heads_[0].bwdl, 3, s, 0.01f);
  scratch_.assign(kH1 + kH1 * 2 + kH2, 0.0f);
}
const HeadBucket& GroupedMlpSmall::headFor(int phase) const {
  if (heads_.size() > 1) {
    int b = phase;
    if (b < 0) b = 0;
    if (b > 2) b = 2;
    return heads_[static_cast<size_t>(b)];
  }
  return heads_[0];
}
void GroupedMlpSmall::forward(const float* tokens, float& value, float* wdl, int phase) const {
  const HeadBucket& h = headFor(phase);
  int h1n = static_cast<int>(h.b1.size());
  int h2n = static_cast<int>(h.b2.size());
  bool isPair = (h.w2.size() == static_cast<size_t>(h2n) * static_cast<size_t>(h1n) * 2);
  if (isPair) {
    float* pre = scratch_.data();
    float* h1p = scratch_.data() + h1n;
    float* h2 = scratch_.data() + h1n + h1n * 2;
    simd::matVec(h.w1.data(), tokens, h.b1.data(), pre, h1n, kIn);
    for (int i = 0; i < h1n; ++i) {
      float c = pre[i] < 0.0f ? 0.0f : (pre[i] > 1.0f ? 1.0f : pre[i]);
      h1p[i] = c;
      h1p[h1n + i] = c * c;
    }
    simd::matVecClipped(h.w2.data(), h1p, h.b2.data(), h2, h2n, h1n * 2);
    float v = h.bvo[0];
    for (int i = 0; i < h2n; ++i) v += h.wvo[i] * h2[i];
    value = std::tanh(v);
    simd::matVec(h.wwdl.data(), h2, h.bwdl.data(), wdl, 3, h2n);
    return;
  }
  float* h1 = scratch_.data();
  float* h2 = scratch_.data() + h1n;
  simd::matVecClipped(h.w1.data(), tokens, h.b1.data(), h1, h1n, kIn);
  simd::matVecClipped(h.w2.data(), h1, h.b2.data(), h2, h2n, h1n);
  float v = h.bvo[0];
  for (int i = 0; i < h2n; ++i) v += h.wvo[i] * h2[i];
  value = std::tanh(v);
  simd::matVec(h.wwdl.data(), h2, h.bwdl.data(), wdl, 3, h2n);
}
size_t GroupedMlpSmall::parameterCount() const {
  size_t n = 0;
  for (const HeadBucket& h : heads_) {
    n += h.w1.size() + h.b1.size() + h.w2.size() + h.b2.size() + h.wvo.size() +
         h.bvo.size() + h.wwdl.size() + h.bwdl.size();
  }
  return n;
}
size_t GroupedMlpSmall::modelSizeBytes() const { return parameterCount() * 4; }
void GroupedMlpSmall::getTensors(std::vector<std::string>& names,
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
    int h1n = static_cast<int>(h.b1.size());
    int h2n = static_cast<int>(h.b2.size());
    int w2cols = h2n == 0 ? 0 : static_cast<int>(h.w2.size() / static_cast<size_t>(h2n));
    shapes.insert(shapes.end(), {{h1n, kIn}, {h1n}, {h2n, w2cols}, {h2n}, {1, h2n}, {1}, {3, h2n}, {3}});
    data.insert(data.end(), {h.w1.data(), h.b1.data(), h.w2.data(), h.b2.data(), h.wvo.data(),
                             h.bvo.data(), h.wwdl.data(), h.bwdl.data()});
  }
}
bool GroupedMlpSmall::setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) {
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
  size_t singlePer = kH1 * kIn + kH1 + kH2 * kH1 + kH2 + kH2 + 1 + 3 * kH2 + 3;
  size_t pairPer = kH1 * kIn + kH1 + kH2 * kH1 * 2 + kH2 + kH2 + 1 + 3 * kH2 + 3;
  bool isPair = (flat.size() == pairPer * nbuckets);
  bool isSingle = (flat.size() == singlePer * nbuckets);
  if (!isPair && !isSingle) return false;
  size_t w2n = isPair ? kH2 * kH1 * 2 : kH2 * kH1;
  heads_.clear();
  size_t off = 0;
  for (size_t b = 0; b < nbuckets; ++b) {
    HeadBucket h;
    h.w1.assign(kH1 * kIn, 0.0f);
    h.b1.assign(kH1, 0.0f);
    h.w2.assign(w2n, 0.0f);
    h.b2.assign(kH2, 0.0f);
    h.wvo.assign(kH2, 0.0f);
    h.bvo.assign(1, 0.0f);
    h.wwdl.assign(3 * kH2, 0.0f);
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
ModelSpec GroupedMlpSmall::spec() const {
  ModelSpec s;
  s.arch = archId();
  bool isPair = !heads_.empty() && heads_[0].w2.size() == heads_[0].b2.size() * heads_[0].b1.size() * 2;
  s.archVersion = isPair ? "0.2.0" : archVersion();
  s.attention = "none";
  s.geometricBias = "none";
  s.headBuckets = static_cast<int>(heads_.size());
  return s;
}
}
