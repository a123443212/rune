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

#include "core/architectures/attention/soft_attention.h"

#include <cmath>
#include <string>

#include "core/architectures/heads/swiglu_head.h"
#include "core/kernels/policy_kernels.h"

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

constexpr float kSoftScale = 0.17677669529663687f;

}

RuneSoftAttnBlock::RuneSoftAttnBlock() {
  uint64_t s = 7777;
  initVec(wq, 32 * 32, s, 0.08f);
  initVec(bq, 32, s, 0.01f);
  initVec(wk, 32 * 32, s, 0.08f);
  initVec(bk, 32, s, 0.01f);
  initVec(wv, 32 * 32, s, 0.08f);
  initVec(bv, 32, s, 0.01f);
  gab.assign(64, 0.0f);
  scratch_.assign(8 * 32 * 3 + 64 * 3 + 8 * 32, 0.0f);
}

void RuneSoftAttnBlock::forward(const float* x, float* out) const {
  float* q = scratch_.data();
  float* k = scratch_.data() + 256;
  float* v = scratch_.data() + 512;
  float* s = scratch_.data() + 768;
  float* w = scratch_.data() + 832;
  float* y = scratch_.data() + 896;
  for (int t = 0; t < 8; ++t) {
    simd::matVec(wq.data(), x + t * 32, bq.data(), q + t * 32, 32, 32);
    simd::matVec(wk.data(), x + t * 32, bk.data(), k + t * 32, 32, 32);
    simd::matVec(wv.data(), x + t * 32, bv.data(), v + t * 32, 32, 32);
  }
  simd::matMulTT(q, k, s, 8, 8, 32);
  for (int a = 0; a < 8; ++a) {
    float logits[8];
    for (int b = 0; b < 8; ++b) logits[b] = s[a * 8 + b] * kSoftScale + gab[a * 8 + b];
    policy::softmax(logits, w + a * 8, 8);
  }
  simd::matMul(w, v, y, 8, 32, 8);
  for (int i = 0; i < 256; ++i) out[i] = x[i] + y[i];
}

size_t RuneSoftAttnBlock::parameterCount() const {
  return wq.size() + bq.size() + wk.size() + bk.size() + wv.size() + bv.size() + gab.size();
}

RuneSoftAttnModel::RuneSoftAttnModel() {
  uint64_t s = 5150;
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

const HeadBucket& RuneSoftAttnModel::headFor(int phase) const {
  if (heads_.size() > 1) {
    int b = phase;
    if (b < 0) b = 0;
    if (b > 2) b = 2;
    return heads_[static_cast<size_t>(b)];
  }
  return heads_[0];
}

void RuneSoftAttnModel::forward(const float* tokens, float& value, float* wdl, int phase) const {
  const HeadBucket& h = headFor(phase);
  float* mixed = scratch_.data();
  attn.forward(tokens, mixed);
  if (headIsSwiGlu(h)) {
    swigluForward(h, mixed, 256, value, wdl, swi_);
    return;
  }
  int h1n = static_cast<int>(h.b1.size());
  int h2n = static_cast<int>(h.b2.size());
  bool isPair = (h.w2.size() == static_cast<size_t>(h2n) * static_cast<size_t>(h1n) * 2);
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

size_t RuneSoftAttnModel::parameterCount() const {
  size_t n = attn.parameterCount();
  for (const HeadBucket& h : heads_) {
    n += h.w1.size() + h.b1.size() + h.w2.size() + h.b2.size() + h.wvo.size() +
         h.bvo.size() + h.wwdl.size() + h.bwdl.size() + h.wgate.size() + h.bgate.size() +
         h.wup.size() + h.bup.size();
  }
  return n;
}

size_t RuneSoftAttnModel::modelSizeBytes() const { return parameterCount() * 4; }

void RuneSoftAttnModel::getTensors(std::vector<std::string>& names,
                                   std::vector<std::vector<int>>& shapes,
                                   std::vector<const float*>& data) const {
  names = {"wq", "bq", "wk", "bk", "wvv", "bvv", "gab"};
  shapes = {{32, 32}, {32}, {32, 32}, {32}, {32, 32}, {32}, {8, 8}};
  data = {attn.wq.data(), attn.bq.data(), attn.wk.data(), attn.bk.data(), attn.wv.data(),
          attn.bv.data(), attn.gab.data()};
  for (size_t b = 0; b < heads_.size(); ++b) {
    std::string suffix = heads_.size() > 1 ? "_b" + std::to_string(b) : "";
    const HeadBucket& h = heads_[b];
    if (headIsSwiGlu(h)) {
      names.insert(names.end(), {"wgate" + suffix, "bgate" + suffix, "wup" + suffix, "bup" + suffix,
                                 "w2" + suffix, "b2" + suffix, "wvo" + suffix, "bvo" + suffix,
                                 "wwdl" + suffix, "bwdl" + suffix});
      int h1n = static_cast<int>(h.bgate.size());
      int h2n = static_cast<int>(h.b2.size());
      shapes.insert(shapes.end(), {{h1n, 256}, {h1n}, {h1n, 256}, {h1n}, {h2n, h1n}, {h2n},
                                   {1, h2n}, {1}, {3, h2n}, {3}});
      data.insert(data.end(), {h.wgate.data(), h.bgate.data(), h.wup.data(), h.bup.data(),
                               h.w2.data(), h.b2.data(), h.wvo.data(), h.bvo.data(),
                               h.wwdl.data(), h.bwdl.data()});
      continue;
    }
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

bool RuneSoftAttnModel::setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) {
  bool swiglu = !names.empty() && names[7] == "wgate";
  size_t perHead = swiglu ? 10 : 8;
  size_t nbuckets = 1;
  if (names.size() > 7 + perHead) nbuckets = 3;
  std::vector<std::string> want = {"wq", "bq", "wk", "bk", "wvv", "bvv", "gab"};
  for (size_t b = 0; b < nbuckets; ++b) {
    std::string suffix = nbuckets > 1 ? "_b" + std::to_string(b) : "";
    if (swiglu) {
      want.insert(want.end(), {"wgate" + suffix, "bgate" + suffix, "wup" + suffix, "bup" + suffix,
                               "w2" + suffix, "b2" + suffix, "wvo" + suffix, "bvo" + suffix,
                               "wwdl" + suffix, "bwdl" + suffix});
    } else {
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
  size_t mixTotal = 0;
  for (auto* slot : slots) mixTotal += slot->size();
  size_t remain = flat.size() > mixTotal ? flat.size() - mixTotal : 0;
  size_t singlePer = 128 * 256 + 128 + 32 * 128 + 32 + 32 + 1 + 3 * 32 + 3;
  size_t pairPer = 128 * 256 + 128 + 32 * 256 + 32 + 32 + 1 + 3 * 32 + 3;
  size_t swigluPer = 128 * 256 + 128 + 128 * 256 + 128 + 32 * 128 + 32 + 32 + 1 + 3 * 32 + 3;
  bool isSwi = swiglu && (remain == swigluPer * nbuckets);
  bool isPair = !swiglu && (remain == pairPer * nbuckets);
  bool isSingle = !swiglu && (remain == singlePer * nbuckets);
  if (!isSwi && !isPair && !isSingle) return false;
  heads_.clear();
  for (size_t b = 0; b < nbuckets; ++b) {
    HeadBucket h;
    if (isSwi) {
      h.wgate.assign(128 * 256, 0.0f);
      h.bgate.assign(128, 0.0f);
      h.wup.assign(128 * 256, 0.0f);
      h.bup.assign(128, 0.0f);
      h.w2.assign(32 * 128, 0.0f);
      h.b2.assign(32, 0.0f);
      h.wvo.assign(32, 0.0f);
      h.bvo.assign(1, 0.0f);
      h.wwdl.assign(3 * 32, 0.0f);
      h.bwdl.assign(3, 0.0f);
      std::vector<std::vector<float>*> hs = {&h.wgate, &h.bgate, &h.wup, &h.bup, &h.w2, &h.b2,
                                             &h.wvo, &h.bvo, &h.wwdl, &h.bwdl};
      for (auto* slot : hs) {
        if (off + slot->size() > flat.size()) return false;
        for (size_t kk = 0; kk < slot->size(); ++kk) (*slot)[kk] = flat[off + kk];
        off += slot->size();
      }
    } else {
      size_t w2n = isPair ? 32 * 256 : 32 * 128;
      h.w1.assign(128 * 256, 0.0f);
      h.b1.assign(128, 0.0f);
      h.w2.assign(w2n, 0.0f);
      h.b2.assign(32, 0.0f);
      h.wvo.assign(32, 0.0f);
      h.bvo.assign(1, 0.0f);
      h.wwdl.assign(3 * 32, 0.0f);
      h.bwdl.assign(3, 0.0f);
      std::vector<std::vector<float>*> hs = {&h.w1, &h.b1, &h.w2, &h.b2, &h.wvo, &h.bvo,
                                             &h.wwdl, &h.bwdl};
      for (auto* slot : hs) {
        if (off + slot->size() > flat.size()) return false;
        for (size_t kk = 0; kk < slot->size(); ++kk) (*slot)[kk] = flat[off + kk];
        off += slot->size();
      }
    }
    heads_.push_back(std::move(h));
  }
  return off == flat.size();
}

ModelSpec RuneSoftAttnModel::spec() const {
  ModelSpec s;
  s.arch = archId();
  bool swi = !heads_.empty() && headIsSwiGlu(heads_[0]);
  bool isPair = !swi && !heads_.empty() &&
                heads_[0].w2.size() == heads_[0].b2.size() * heads_[0].b1.size() * 2;
  s.archVersion = swi ? "0.3.0" : (isPair ? "0.2.0" : archVersion());
  s.attention = "softmax_scaled";
  s.geometricBias = "learned";
  s.head = swi ? "value_swiglu" : (isPair ? "value_wdl_pair" : "value_wdl");
  s.gate = "softmax";
  s.headBuckets = static_cast<int>(heads_.size());
  return s;
}

}
