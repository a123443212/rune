#include "core/architectures/attention/attention.h"

#include <cmath>

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

RuneAttentionBlock::RuneAttentionBlock(bool useGab) : useGab_(useGab) {
  uint64_t s = useGab ? 999 : 555;
  initVec(wq, 32 * 32, s, 0.08f);
  initVec(bq, 32, s, 0.01f);
  initVec(wk, 32 * 32, s, 0.08f);
  initVec(bk, 32, s, 0.01f);
  initVec(wv, 32 * 32, s, 0.08f);
  initVec(bv, 32, s, 0.01f);
  gab.assign(64, 0.0f);
  scratch_.assign(8 * 32 * 3 + 64 + 8 * 32, 0.0f);
}

void RuneAttentionBlock::forward(const float* x, float* out) const {
  float* q = scratch_.data();
  float* k = scratch_.data() + 256;
  float* v = scratch_.data() + 512;
  float* s = scratch_.data() + 768;
  float* y = scratch_.data() + 832;
  for (int t = 0; t < 8; ++t) {
    simd::matVec(wq.data(), x + t * 32, bq.data(), q + t * 32, 32, 32);
    simd::matVec(wk.data(), x + t * 32, bk.data(), k + t * 32, 32, 32);
    simd::matVec(wv.data(), x + t * 32, bv.data(), v + t * 32, 32, 32);
  }
  simd::matMulTT(q, k, s, 8, 8, 32);
  for (int i = 0; i < 64; ++i) {
    float val = s[i] + (useGab_ ? gab[i] : 0.0f);
    s[i] = simd::clippedRelu(val);
  }
  simd::matMul(s, v, y, 8, 32, 8);
  for (int i = 0; i < 256; ++i) out[i] = x[i] + y[i];
}

size_t RuneAttentionBlock::parameterCount() const {
  size_t n = wq.size() + bq.size() + wk.size() + bk.size() + wv.size() + bv.size();
  if (useGab_) n += gab.size();
  return n;
}

RuneAttnModel::RuneAttnModel() : RuneAttnModel(false) {}

RuneAttnModel::RuneAttnModel(bool useGab) : attn(useGab), useGab_(useGab) {
  uint64_t s = useGab ? 4242 : 3131;
  initVec(w1, 128 * 256, s, 0.05f);
  initVec(b1, 128, s, 0.01f);
  initVec(w2, 32 * 128, s, 0.05f);
  initVec(b2, 32, s, 0.01f);
  initVec(wvo, 1 * 32, s, 0.05f);
  initVec(bvo, 1, s, 0.01f);
  initVec(wwdl, 3 * 32, s, 0.05f);
  initVec(bwdl, 3, s, 0.01f);
  scratch_.assign(256 + 128 + 32, 0.0f);
}

void RuneAttnModel::forward(const float* tokens, float& value, float* wdl) const {
  float* mixed = scratch_.data();
  float* h1 = scratch_.data() + 256;
  float* h2 = scratch_.data() + 256 + 128;
  attn.forward(tokens, mixed);
  simd::matVecClipped(w1.data(), mixed, b1.data(), h1, 128, 256);
  simd::matVecClipped(w2.data(), h1, b2.data(), h2, 32, 128);
  float vv = bvo[0];
  for (int i = 0; i < 32; ++i) vv += wvo[i] * h2[i];
  value = std::tanh(vv);
  simd::matVec(wwdl.data(), h2, bwdl.data(), wdl, 3, 32);
}

size_t RuneAttnModel::parameterCount() const {
  return attn.parameterCount() + w1.size() + b1.size() + w2.size() + b2.size() + wvo.size() +
         bvo.size() + wwdl.size() + bwdl.size();
}

size_t RuneAttnModel::modelSizeBytes() const { return parameterCount() * 4; }

void RuneAttnModel::getTensors(std::vector<std::string>& names,
                               std::vector<std::vector<int>>& shapes,
                               std::vector<const float*>& data) const {
  names = {"wq", "bq", "wk", "bk", "wvv", "bvv", "gab", "w1", "b1", "w2", "b2", "wvo", "bvo",
           "wwdl", "bwdl"};
  shapes = {{32, 32}, {32}, {32, 32}, {32}, {32, 32}, {32}, {8, 8}, {128, 256}, {128},
            {32, 128}, {32}, {1, 32}, {1}, {3, 32}, {3}};
  data = {attn.wq.data(), attn.bq.data(), attn.wk.data(), attn.bk.data(), attn.wv.data(),
          attn.bv.data(), attn.gab.data(), w1.data(), b1.data(), w2.data(), b2.data(),
          wvo.data(), bvo.data(), wwdl.data(), bwdl.data()};
}

bool RuneAttnModel::setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) {
  std::vector<std::string> want = {"wq", "bq", "wk", "bk", "wvv", "bvv", "gab", "w1", "b1",
                                   "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"};
  if (names.size() != want.size()) return false;
  for (size_t i = 0; i < want.size(); ++i) {
    if (names[i] != want[i]) return false;
  }
  std::vector<std::vector<float>*> slots = {&attn.wq, &attn.bq, &attn.wk, &attn.bk, &attn.wv,
                                            &attn.bv, &attn.gab, &w1, &b1, &w2, &b2, &wvo,
                                            &bvo, &wwdl, &bwdl};
  size_t off = 0;
  for (auto* slot : slots) {
    if (off + slot->size() > flat.size()) return false;
    for (size_t kk = 0; kk < slot->size(); ++kk) (*slot)[kk] = flat[off + kk];
    off += slot->size();
  }
  return off == flat.size();
}

ModelSpec RuneAttnModel::spec() const {
  ModelSpec s;
  s.arch = archId();
  s.archVersion = archVersion();
  s.attention = "gated_linear";
  s.geometricBias = useGab_ ? "learned" : "none";
  return s;
}

}
