#include "core/architectures/mlp/mlp.h"

#include <cmath>
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
  initVec(w1, 128 * 256, s, 0.05f);
  initVec(b1, 128, s, 0.01f);
  initVec(w2, 32 * 128, s, 0.05f);
  initVec(b2, 32, s, 0.01f);
  initVec(wv, 1 * 32, s, 0.05f);
  initVec(bv, 1, s, 0.01f);
  initVec(wwdl, 3 * 32, s, 0.05f);
  initVec(bwdl, 3, s, 0.01f);
  scratch_.assign(160, 0.0f);
}

void GroupedMlp::forward(const float* tokens, float& value, float* wdl) const {
  float* h1 = scratch_.data();
  float* h2 = scratch_.data() + 128;
  simd::matVecClipped(w1.data(), tokens, b1.data(), h1, kH1, kIn);
  simd::matVecClipped(w2.data(), h1, b2.data(), h2, kH2, kH1);
  float v = bv[0];
  for (int i = 0; i < kH2; ++i) v += wv[i] * h2[i];
  value = std::tanh(v);
  simd::matVec(wwdl.data(), h2, bwdl.data(), wdl, 3, kH2);
}

size_t GroupedMlp::parameterCount() const {
  return w1.size() + b1.size() + w2.size() + b2.size() + wv.size() + bv.size() + wwdl.size() +
         bwdl.size();
}

size_t GroupedMlp::modelSizeBytes() const { return parameterCount() * 4; }

void GroupedMlp::getTensors(std::vector<std::string>& names, std::vector<std::vector<int>>& shapes,
                            std::vector<const float*>& data) const {
  names = {"w1", "b1", "w2", "b2", "wv", "bv", "wwdl", "bwdl"};
  shapes = {{128, 256}, {128}, {32, 128}, {32}, {1, 32}, {1}, {3, 32}, {3}};
  data = {w1.data(), b1.data(), w2.data(), b2.data(), wv.data(), bv.data(), wwdl.data(),
          bwdl.data()};
}

bool GroupedMlp::setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) {
  std::vector<std::vector<float>*> slots = {&w1, &b1, &w2, &b2, &wv, &bv, &wwdl, &bwdl};
  std::vector<std::string> want = {"w1", "b1", "w2", "b2", "wv", "bv", "wwdl", "bwdl"};
  if (names.size() != want.size()) return false;
  for (size_t i = 0; i < want.size(); ++i) {
    if (names[i] != want[i]) return false;
  }
  size_t off = 0;
  for (auto* slot : slots) {
    if (off + slot->size() > flat.size()) return false;
    for (size_t k = 0; k < slot->size(); ++k) (*slot)[k] = flat[off + k];
    off += slot->size();
  }
  return off == flat.size();
}

ModelSpec GroupedMlp::spec() const {
  ModelSpec s;
  s.arch = archId();
  s.archVersion = archVersion();
  s.attention = "none";
  s.geometricBias = "none";
  return s;
}

SfnnBaseline::SfnnBaseline() {
  uint64_t s = 777;
  initVec(w1, 256 * 256, s, 0.04f);
  initVec(b1, 256, s, 0.01f);
  initVec(w2, 32 * 256, s, 0.05f);
  initVec(b2, 32, s, 0.01f);
  initVec(wv, 1 * 32, s, 0.05f);
  initVec(bv, 1, s, 0.01f);
  initVec(wwdl, 3 * 32, s, 0.05f);
  initVec(bwdl, 3, s, 0.01f);
  scratch_.assign(288, 0.0f);
}

void SfnnBaseline::forward(const float* tokens, float& value, float* wdl) const {
  float* h1 = scratch_.data();
  float* h2 = scratch_.data() + 256;
  simd::matVecClipped(w1.data(), tokens, b1.data(), h1, kH1, kIn);
  simd::matVecClipped(w2.data(), h1, b2.data(), h2, kH2, kH1);
  float v = bv[0];
  for (int i = 0; i < kH2; ++i) v += wv[i] * h2[i];
  value = std::tanh(v);
  simd::matVec(wwdl.data(), h2, bwdl.data(), wdl, 3, kH2);
}

size_t SfnnBaseline::parameterCount() const {
  return w1.size() + b1.size() + w2.size() + b2.size() + wv.size() + bv.size() + wwdl.size() +
         bwdl.size();
}

size_t SfnnBaseline::modelSizeBytes() const { return parameterCount() * 4; }

void SfnnBaseline::getTensors(std::vector<std::string>& names, std::vector<std::vector<int>>& shapes,
                              std::vector<const float*>& data) const {
  names = {"w1", "b1", "w2", "b2", "wv", "bv", "wwdl", "bwdl"};
  shapes = {{256, 256}, {256}, {32, 256}, {32}, {1, 32}, {1}, {3, 32}, {3}};
  data = {w1.data(), b1.data(), w2.data(), b2.data(), wv.data(), bv.data(), wwdl.data(),
          bwdl.data()};
}

bool SfnnBaseline::setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) {
  std::vector<std::vector<float>*> slots = {&w1, &b1, &w2, &b2, &wv, &bv, &wwdl, &bwdl};
  std::vector<std::string> want = {"w1", "b1", "w2", "b2", "wv", "bv", "wwdl", "bwdl"};
  if (names.size() != want.size()) return false;
  for (size_t i = 0; i < want.size(); ++i) {
    if (names[i] != want[i]) return false;
  }
  size_t off = 0;
  for (auto* slot : slots) {
    if (off + slot->size() > flat.size()) return false;
    for (size_t k = 0; k < slot->size(); ++k) (*slot)[k] = flat[off + k];
    off += slot->size();
  }
  return off == flat.size();
}

ModelSpec SfnnBaseline::spec() const {
  ModelSpec s;
  s.arch = archId();
  s.archVersion = archVersion();
  s.attention = "none";
  s.geometricBias = "none";
  return s;
}

}
