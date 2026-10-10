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

#include "core/architectures/attention/attention.h"

#include <cmath>
#include <string>

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

RuneAttentionBlock::RuneAttentionBlock(bool useGab, GateFn gate) : useGab_(useGab), gate_(gate) {
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
    s[i] = applyGate(gate_, val);
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

RuneAttnModel::RuneAttnModel(bool useGab, GateFn gate) : attn(useGab, gate), useGab_(useGab), gate_(gate) {
  uint64_t s = useGab ? 4242 : 3131;
  heads_.resize(1);
  initVec(heads_[0].w1, 128 * 256, s, 0.05f);
  initVec(heads_[0].b1, 128, s, 0.01f);
  initVec(heads_[0].w2, 32 * 128, s, 0.05f);
  initVec(heads_[0].b2, 32, s, 0.01f);
  initVec(heads_[0].wvo, 1 * 32, s, 0.05f);
  initVec(heads_[0].bvo, 1, s, 0.01f);
  initVec(heads_[0].wwdl, 3 * 32, s, 0.05f);
  initVec(heads_[0].bwdl, 3, s, 0.01f);
  scratch_.assign(256 + 128 + 256 + 32, 0.0f);
}

const HeadBucket& RuneAttnModel::headFor(int phase) const {
  if (heads_.size() > 1) {
    int b = phase;
    if (b < 0) b = 0;
    if (b > 2) b = 2;
    return heads_[static_cast<size_t>(b)];
  }
  return heads_[0];
}

void RuneAttnModel::forward(const float* tokens, float& value, float* wdl, int phase) const {
  const HeadBucket& h = headFor(phase);
  int h1n = static_cast<int>(h.b1.size());
  int h2n = static_cast<int>(h.b2.size());
  bool isPair = (h.w2.size() == static_cast<size_t>(h2n) * static_cast<size_t>(h1n) * 2);
  float* mixed = scratch_.data();
  attn.forward(tokens, mixed);
  if (isPair) {
    float* pre = scratch_.data() + 256;
    float* h1p = scratch_.data() + 256 + h1n;
    float* h2 = scratch_.data() + 256 + h1n + h1n * 2;
    simd::matVec(h.w1.data(), mixed, h.b1.data(), pre, h1n, 256);
    for (int i = 0; i < h1n; ++i) {
      float c = pre[i] < 0.0f ? 0.0f : (pre[i] > 1.0f ? 1.0f : pre[i]);
      h1p[i] = c;
      h1p[h1n + i] = c * c;
    }
    simd::matVecClipped(h.w2.data(), h1p, h.b2.data(), h2, h2n, h1n * 2);
    float vv = h.bvo[0];
    for (int i = 0; i < h2n; ++i) vv += h.wvo[i] * h2[i];
    value = std::tanh(vv);
    simd::matVec(h.wwdl.data(), h2, h.bwdl.data(), wdl, 3, h2n);
    return;
  }
  float* h1 = scratch_.data() + 256;
  float* h2 = scratch_.data() + 256 + h1n;
  simd::matVecClipped(h.w1.data(), mixed, h.b1.data(), h1, h1n, 256);
  simd::matVecClipped(h.w2.data(), h1, h.b2.data(), h2, h2n, h1n);
  float vv = h.bvo[0];
  for (int i = 0; i < h2n; ++i) vv += h.wvo[i] * h2[i];
  value = std::tanh(vv);
  simd::matVec(h.wwdl.data(), h2, h.bwdl.data(), wdl, 3, h2n);
}

size_t RuneAttnModel::parameterCount() const {
  size_t n = attn.parameterCount();
  for (const HeadBucket& h : heads_) {
    n += h.w1.size() + h.b1.size() + h.w2.size() + h.b2.size() + h.wvo.size() +
         h.bvo.size() + h.wwdl.size() + h.bwdl.size();
  }
  return n;
}

size_t RuneAttnModel::modelSizeBytes() const { return parameterCount() * 4; }

void RuneAttnModel::getTensors(std::vector<std::string>& names,
                               std::vector<std::vector<int>>& shapes,
                               std::vector<const float*>& data) const {
  names = {"wq", "bq", "wk", "bk", "wvv", "bvv", "gab"};
  shapes = {{32, 32}, {32}, {32, 32}, {32}, {32, 32}, {32}, {8, 8}};
  data = {attn.wq.data(), attn.bq.data(), attn.wk.data(), attn.bk.data(), attn.wv.data(),
          attn.bv.data(), attn.gab.data()};
  for (size_t b = 0; b < heads_.size(); ++b) {
    std::string suffix = heads_.size() > 1 ? "_b" + std::to_string(b) : "";
    const HeadBucket& h = heads_[b];
    names.insert(names.end(), {"w1" + suffix, "b1" + suffix, "w2" + suffix, "b2" + suffix,
                               "wvo" + suffix, "bvo" + suffix, "wwdl" + suffix, "bwdl" + suffix});
    int h1n = static_cast<int>(h.b1.size());
    int h2n = static_cast<int>(h.b2.size());
    int w2cols = h2n == 0 ? 0 : static_cast<int>(h.w2.size() / static_cast<size_t>(h2n));
    shapes.insert(shapes.end(), {{h1n, 256}, {h1n}, {h2n, w2cols}, {h2n}, {1, h2n}, {1}, {3, h2n}, {3}});
    data.insert(data.end(), {h.w1.data(), h.b1.data(), h.w2.data(), h.b2.data(), h.wvo.data(),
                             h.bvo.data(), h.wwdl.data(), h.bwdl.data()});
  }
}

bool RuneAttnModel::setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) {
  std::vector<std::string> base = {"wq", "bq", "wk", "bk", "wvv", "bvv", "gab", "w1", "b1",
                                   "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"};
  std::vector<std::string> want = base;
  if (names.size() > base.size()) {
    want.clear();
    want.insert(want.end(), base.begin(), base.begin() + 7);
    for (int b = 0; b < 3; ++b) {
      std::string suffix = "_b" + std::to_string(b);
      want.insert(want.end(), {"w1" + suffix, "b1" + suffix, "w2" + suffix, "b2" + suffix,
                               "wvo" + suffix, "bvo" + suffix, "wwdl" + suffix, "bwdl" + suffix});
    }
  }
  if (names.size() != want.size()) return false;
  for (size_t i = 0; i < want.size(); ++i) {
    if (names[i] != want[i]) return false;
  }
  std::vector<std::vector<float>*> slots = {&attn.wq, &attn.bq, &attn.wk, &attn.bk, &attn.wv,
                                            &attn.bv, &attn.gab};
  size_t off = 0;
  for (auto* slot : slots) {
    if (off + slot->size() > flat.size()) return false;
    for (size_t kk = 0; kk < slot->size(); ++kk) (*slot)[kk] = flat[off + kk];
    off += slot->size();
  }
  size_t nbuckets = (want.size() - 7) / 8;
  size_t mixTotal = 0;
  for (auto* slot : slots) mixTotal += slot->size();
  size_t remain = flat.size() > mixTotal ? flat.size() - mixTotal : 0;
  size_t singlePer = 128 * 256 + 128 + 32 * 128 + 32 + 32 + 1 + 3 * 32 + 3;
  size_t pairPer = 128 * 256 + 128 + 32 * 256 + 32 + 32 + 1 + 3 * 32 + 3;
  bool isPair = (remain == pairPer * nbuckets);
  bool isSingle = (remain == singlePer * nbuckets);
  if (!isPair && !isSingle) return false;
  size_t w2n = isPair ? 32 * 256 : 32 * 128;
  heads_.clear();
  for (size_t b = 0; b < nbuckets; ++b) {
    HeadBucket h;
    h.w1.assign(128 * 256, 0.0f);
    h.b1.assign(128, 0.0f);
    h.w2.assign(w2n, 0.0f);
    h.b2.assign(32, 0.0f);
    h.wvo.assign(32, 0.0f);
    h.bvo.assign(1, 0.0f);
    h.wwdl.assign(3 * 32, 0.0f);
    h.bwdl.assign(3, 0.0f);
    std::vector<std::vector<float>*> hs = {&h.w1, &h.b1, &h.w2, &h.b2, &h.wvo, &h.bvo, &h.wwdl, &h.bwdl};
    for (auto* slot : hs) {
      if (off + slot->size() > flat.size()) return false;
      for (size_t kk = 0; kk < slot->size(); ++kk) (*slot)[kk] = flat[off + kk];
      off += slot->size();
    }
    heads_.push_back(std::move(h));
  }
  return off == flat.size();
}

ModelSpec RuneAttnModel::spec() const {
  ModelSpec s;
  s.arch = archId();
  bool isPair = !heads_.empty() && heads_[0].w2.size() == heads_[0].b2.size() * heads_[0].b1.size() * 2;
  s.archVersion = isPair ? "0.2.0" : archVersion();
  s.attention = "gated_linear";
  s.geometricBias = useGab_ ? "learned" : "none";
  s.gate = gateName(gate_);
  s.headBuckets = static_cast<int>(heads_.size());
  return s;
}

}
