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

#include "core/architectures/relational/relational.h"

#include <cmath>

#include "core/architectures/heads/swiglu_head.h"

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

RelationalMixer::RelationalMixer() {
  RelationalConfig cfg;
  std::string err;
  configure(cfg, err);
}

bool RelationalMixer::configure(const RelationalConfig& cfg, std::string& err) {
  if (cfg.tokens != 6 && cfg.tokens != 8 && cfg.tokens != 10) {
    err = "bad token count";
    return false;
  }
  if (cfg.dim != 24 && cfg.dim != 32 && cfg.dim != 40) {
    err = "bad token dim";
    return false;
  }
  cfg_ = cfg;
  int t = cfg.tokens;
  int d = cfg.dim;
  uint64_t s = 20240 + t * 100 + d;
  initVec(wq, d * d, s, 0.08f);
  initVec(bq, d, s, 0.01f);
  initVec(wk, d * d, s, 0.08f);
  initVec(bk, d, s, 0.01f);
  initVec(wv, d * d, s, 0.08f);
  initVec(bv, d, s, 0.01f);
  gabS.assign(t * t, 0.0f);
  dynU.assign(t * ContextSpec::kDim, 0.0f);
  dynW.assign(t * ContextSpec::kDim, 0.0f);
  scratch_.assign(t * d * 3 + t * t * 2 + t * d, 0.0f);
  return true;
}

void RelationalMixer::forward(const float* x, const float* ctx, float* out) const {
  int t = cfg_.tokens;
  int d = cfg_.dim;
  float* q = scratch_.data();
  float* k = scratch_.data() + t * d;
  float* v = scratch_.data() + 2 * t * d;
  float* s = scratch_.data() + 3 * t * d;
  float* y = scratch_.data() + 3 * t * d + t * t;
  for (int i = 0; i < t; ++i) {
    simd::matVec(wq.data(), x + i * d, bq.data(), q + i * d, d, d);
    simd::matVec(wk.data(), x + i * d, bk.data(), k + i * d, d, d);
    simd::matVec(wv.data(), x + i * d, bv.data(), v + i * d, d, d);
  }
  simd::matMulTT(q, k, s, t, t, d);
  for (int a = 0; a < t; ++a) {
    for (int b = 0; b < t; ++b) {
      float val = s[a * t + b] + gabS[a * t + b];
      if (cfg_.dynamicBias && ctx) {
        float u = 0.0f;
        float w = 0.0f;
        for (int c = 0; c < ContextSpec::kDim; ++c) {
          u += dynU[a * ContextSpec::kDim + c] * ctx[c];
          w += dynW[b * ContextSpec::kDim + c] * ctx[c];
        }
        float delta = u * w;
        if (delta < -0.25f) delta = -0.25f;
        if (delta > 0.25f) delta = 0.25f;
        val += delta;
      }
      s[a * t + b] = applyGate(cfg_.gate, val);
    }
  }
  simd::matMul(s, v, y, t, d, t);
  for (int i = 0; i < t * d; ++i) out[i] = x[i] + cfg_.alpha * y[i];
}

size_t RelationalMixer::parameterCount() const {
  size_t n = wq.size() + bq.size() + wk.size() + bk.size() + wv.size() + bv.size() + gabS.size();
  if (cfg_.dynamicBias) n += dynU.size() + dynW.size();
  return n;
}

RelationalModel::RelationalModel() {
  RelationalConfig cfg;
  std::string err;
  configure(cfg, err);
}

bool RelationalModel::configure(const RelationalConfig& cfg, std::string& err) {
  if (!mixer.configure(cfg, err)) return false;
  int in = cfg.tokens * cfg.dim;
  uint64_t s = 5150 + cfg.tokens * 100 + cfg.dim;
  heads_.resize(1);
  initVec(heads_[0].w1, 128 * in, s, 0.05f);
  initVec(heads_[0].b1, 128, s, 0.01f);
  initVec(heads_[0].w2, 32 * 128, s, 0.05f);
  initVec(heads_[0].b2, 32, s, 0.01f);
  initVec(heads_[0].wvo, 1 * 32, s, 0.05f);
  initVec(heads_[0].bvo, 1, s, 0.01f);
  initVec(heads_[0].wwdl, 3 * 32, s, 0.05f);
  initVec(heads_[0].bwdl, 3, s, 0.01f);
  scratch_.assign(in + 128 + 256 + 32, 0.0f);
  return true;
}

const HeadBucket& RelationalModel::headFor(int phase) const {
  if (heads_.size() > 1) {
    int b = phase;
    if (b < 0) b = 0;
    if (b > 2) b = 2;
    return heads_[static_cast<size_t>(b)];
  }
  return heads_[0];
}

void RelationalModel::forward(const float* tokens, float& value, float* wdl, int phase) const {
  float ctx[ContextSpec::kDim] = {};
  forwardWithContext(tokens, ctx, value, wdl, phase);
}

void RelationalModel::forwardWithContext(const float* tokens, const float* ctx, float& value,
                                         float* wdl, int phase) const {
  const HeadBucket& h = headFor(phase);
  int in = mixer.config().tokens * mixer.config().dim;
  int h1n = static_cast<int>(h.b1.size());
  int h2n = static_cast<int>(h.b2.size());
  bool isPair = (h.w2.size() == static_cast<size_t>(h2n) * static_cast<size_t>(h1n) * 2);
  float* mixed = scratch_.data();
  mixer.forward(tokens, ctx, mixed);
  if (headIsSwiGlu(h)) {
    swigluForward(h, mixed, in, value, wdl, swi_);
    return;
  }
  if (isPair) {
    float* pre = scratch_.data() + in;
    float* h1p = scratch_.data() + in + h1n;
    float* h2 = scratch_.data() + in + h1n + h1n * 2;
    simd::matVec(h.w1.data(), mixed, h.b1.data(), pre, h1n, in);
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
  float* h1 = scratch_.data() + in;
  float* h2 = scratch_.data() + in + 128;
  simd::matVecClipped(h.w1.data(), mixed, h.b1.data(), h1, 128, in);
  simd::matVecClipped(h.w2.data(), h1, h.b2.data(), h2, 32, 128);
  float vv = h.bvo[0];
  for (int i = 0; i < 32; ++i) vv += h.wvo[i] * h2[i];
  value = std::tanh(vv);
  simd::matVec(h.wwdl.data(), h2, h.bwdl.data(), wdl, 3, 32);
}

size_t RelationalModel::parameterCount() const {
  size_t n = mixer.parameterCount();
  for (const HeadBucket& h : heads_) {
    n += h.w1.size() + h.b1.size() + h.w2.size() + h.b2.size() + h.wvo.size() +
         h.bvo.size() + h.wwdl.size() + h.bwdl.size() + h.wgate.size() + h.bgate.size() +
         h.wup.size() + h.bup.size();
  }
  return n;
}

void RelationalModel::getTensors(std::vector<std::string>& names,
                                 std::vector<std::vector<int>>& shapes,
                                 std::vector<const float*>& data) const {
  int t = mixer.config().tokens;
  int d = mixer.config().dim;
  int in = t * d;
  names = {"wq", "bq", "wk", "bk", "wvv", "bvv", "gabS"};
  shapes = {{d, d}, {d}, {d, d}, {d}, {d, d}, {d}, {t, t}};
  data = {mixer.wq.data(), mixer.bq.data(), mixer.wk.data(), mixer.bk.data(), mixer.wv.data(),
          mixer.bv.data(), mixer.gabS.data()};
  if (mixer.config().dynamicBias) {
    names.push_back("dynU");
    names.push_back("dynW");
    shapes.push_back({t, ContextSpec::kDim});
    shapes.push_back({t, ContextSpec::kDim});
    data.push_back(mixer.dynU.data());
    data.push_back(mixer.dynW.data());
  }
  if (heads_.size() > 1) {
    for (size_t b = 0; b < heads_.size(); ++b) {
      std::string suffix = "_b" + std::to_string(b);
      const HeadBucket& h = heads_[b];
      if (headIsSwiGlu(h)) {
        names.insert(names.end(), {"wgate" + suffix, "bgate" + suffix, "wup" + suffix, "bup" + suffix,
                                   "w2" + suffix, "b2" + suffix, "wvo" + suffix, "bvo" + suffix,
                                   "wwdl" + suffix, "bwdl" + suffix});
        int h1nn = static_cast<int>(h.bgate.size());
        int h2nn = static_cast<int>(h.b2.size());
        shapes.insert(shapes.end(), {{h1nn, in}, {h1nn}, {h1nn, in}, {h1nn}, {h2nn, h1nn}, {h2nn},
                                     {1, h2nn}, {1}, {3, h2nn}, {3}});
        data.insert(data.end(), {h.wgate.data(), h.bgate.data(), h.wup.data(), h.bup.data(),
                                 h.w2.data(), h.b2.data(), h.wvo.data(), h.bvo.data(),
                                 h.wwdl.data(), h.bwdl.data()});
        continue;
      }
      names.insert(names.end(), {"w1" + suffix, "b1" + suffix, "w2" + suffix, "b2" + suffix,
                                 "wvo" + suffix, "bvo" + suffix, "wwdl" + suffix, "bwdl" + suffix});
      int h1nn = static_cast<int>(h.b1.size());
      int h2nn = static_cast<int>(h.b2.size());
      int w2c = h2nn == 0 ? 0 : static_cast<int>(h.w2.size() / static_cast<size_t>(h2nn));
      shapes.insert(shapes.end(), {{128, in}, {h1nn}, {h2nn, w2c}, {h2nn}, {1, h2nn}, {1}, {3, h2nn}, {3}});
      data.insert(data.end(), {h.w1.data(), h.b1.data(), h.w2.data(), h.b2.data(), h.wvo.data(),
                               h.bvo.data(), h.wwdl.data(), h.bwdl.data()});
    }
  } else {
    if (headIsSwiGlu(heads_[0])) {
      const HeadBucket& h = heads_[0];
      names.insert(names.end(), {"wgate", "bgate", "wup", "bup", "w2", "b2", "wvo", "bvo",
                                 "wwdl", "bwdl"});
      int h1nn = static_cast<int>(h.bgate.size());
      int h2nn = static_cast<int>(h.b2.size());
      shapes.insert(shapes.end(), {{h1nn, in}, {h1nn}, {h1nn, in}, {h1nn}, {h2nn, h1nn}, {h2nn},
                                   {1, h2nn}, {1}, {3, h2nn}, {3}});
      data.insert(data.end(), {h.wgate.data(), h.bgate.data(), h.wup.data(), h.bup.data(),
                               h.w2.data(), h.b2.data(), h.wvo.data(), h.bvo.data(),
                               h.wwdl.data(), h.bwdl.data()});
    } else {
      names.insert(names.end(), {"w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"});
    int h1nn = static_cast<int>(heads_[0].b1.size());
    int h2nn = static_cast<int>(heads_[0].b2.size());
    int w2c = h2nn == 0 ? 0 : static_cast<int>(heads_[0].w2.size() / static_cast<size_t>(h2nn));
    shapes.insert(shapes.end(), {{128, in}, {h1nn}, {h2nn, w2c}, {h2nn}, {1, h2nn}, {1}, {3, h2nn}, {3}});
    data.insert(data.end(), {heads_[0].w1.data(), heads_[0].b1.data(), heads_[0].w2.data(),
                             heads_[0].b2.data(), heads_[0].wvo.data(), heads_[0].bvo.data(),
                             heads_[0].wwdl.data(), heads_[0].bwdl.data()});
    }
  }
}

bool RelationalModel::setTensors(const std::vector<std::string>& names,
                                 const std::vector<float>& flat) {
  std::vector<std::string> want = {"wq", "bq", "wk", "bk", "wvv", "bvv", "gabS"};
  if (mixer.config().dynamicBias) {
    want.push_back("dynU");
    want.push_back("dynW");
  }
  size_t mixBase = want.size();
  size_t nbuckets = 1;
  bool swiglu = names.size() > mixBase && (names[mixBase] == "wgate" || names[mixBase] == "wgate_b0");
  size_t perHead = swiglu ? 10 : 8;
  std::vector<const char*> headNames = {"w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"};
  if (swiglu) headNames = {"wgate", "bgate", "wup", "bup", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"};
  if (names.size() > mixBase + perHead) {
    nbuckets = 3;
    for (int b = 0; b < 3; ++b) {
      std::string suffix = "_b" + std::to_string(b);
      for (const char* n : headNames) {
        want.push_back(std::string(n) + suffix);
      }
    }
  } else {
    for (const char* n : headNames) want.push_back(n);
  }
  if (names.size() != want.size()) return false;
  for (size_t i = 0; i < want.size(); ++i) {
    if (names[i] != want[i]) return false;
  }
  std::vector<std::vector<float>*> slots = {&mixer.wq, &mixer.bq, &mixer.wk, &mixer.bk,
                                            &mixer.wv, &mixer.bv, &mixer.gabS};
  if (mixer.config().dynamicBias) {
    slots.push_back(&mixer.dynU);
    slots.push_back(&mixer.dynW);
  }
  int in = mixer.config().tokens * mixer.config().dim;
  size_t mixTotal = 0;
  for (auto* s : slots) mixTotal += s->size();
  size_t rem = flat.size() > mixTotal ? flat.size() - mixTotal : 0;
  size_t singlePer = 128 * in + 128 + 32 * 128 + 32 + 32 + 1 + 3 * 32 + 3;
  size_t pairPer = 128 * in + 128 + 32 * 256 + 32 + 32 + 1 + 3 * 32 + 3;
  size_t swigluPer = 128 * in + 128 + 128 * in + 128 + 32 * 128 + 32 + 32 + 1 + 3 * 32 + 3;
  bool isSwi = swiglu && (rem == swigluPer * nbuckets);
  bool isP = !swiglu && (rem == pairPer * nbuckets);
  bool isS = !swiglu && (rem == singlePer * nbuckets);
  if (!isSwi && !isP && !isS) return false;
  size_t w2nn = isP ? 32 * 256 : 32 * 128;
  heads_.clear();
  for (size_t b = 0; b < nbuckets; ++b) {
    HeadBucket h;
    if (isSwi) {
      h.wgate.assign(128 * in, 0.0f);
      h.bgate.assign(128, 0.0f);
      h.wup.assign(128 * in, 0.0f);
      h.bup.assign(128, 0.0f);
      h.w2.assign(32 * 128, 0.0f);
      h.b2.assign(32, 0.0f);
      h.wvo.assign(32, 0.0f);
      h.bvo.assign(1, 0.0f);
      h.wwdl.assign(3 * 32, 0.0f);
      h.bwdl.assign(3, 0.0f);
    } else {
      h.w1.assign(128 * in, 0.0f);
      h.b1.assign(128, 0.0f);
      h.w2.assign(w2nn, 0.0f);
      h.b2.assign(32, 0.0f);
      h.wvo.assign(32, 0.0f);
      h.bvo.assign(1, 0.0f);
      h.wwdl.assign(3 * 32, 0.0f);
      h.bwdl.assign(3, 0.0f);
    }
    heads_.push_back(std::move(h));
  }
  for (size_t b = 0; b < nbuckets; ++b) {
    HeadBucket& h = heads_[b];
    std::vector<std::vector<float>*> hs;
    if (isSwi) {
      hs = {&h.wgate, &h.bgate, &h.wup, &h.bup, &h.w2, &h.b2,
            &h.wvo, &h.bvo, &h.wwdl, &h.bwdl};
    } else {
      hs = {&h.w1, &h.b1, &h.w2, &h.b2, &h.wvo, &h.bvo, &h.wwdl, &h.bwdl};
    }
    for (auto* slot : hs) slots.push_back(slot);
  }
  size_t off = 0;
  for (auto* slot : slots) {
    if (off + slot->size() > flat.size()) return false;
    for (size_t k = 0; k < slot->size(); ++k) (*slot)[k] = flat[off + k];
    off += slot->size();
  }
  return off == flat.size();
}

ModelSpec RelationalModel::spec() const {
  ModelSpec s;
  s.arch = archId();
  bool swi = !heads_.empty() && headIsSwiGlu(heads_[0]);
  bool isPair = !swi && !heads_.empty() &&
                heads_[0].w2.size() == heads_[0].b2.size() * heads_[0].b1.size() * 2;
  s.archVersion = swi ? "0.3.0" : (isPair ? "0.2.1" : archVersion());
  s.head = swi ? "value_swiglu" : (isPair ? "value_wdl_pair" : "value_wdl");
  s.tokens = mixer.config().tokens;
  s.tokenDim = mixer.config().dim;
  s.attention = "gated_relational";
  s.geometricBias = mixer.config().dynamicBias ? "dynamic" : "static";
  s.gate = gateName(mixer.config().gate);
  s.alpha = mixer.config().alpha;
  s.headBuckets = static_cast<int>(heads_.size());
  return s;
}

RelationalEvaluator::RelationalEvaluator() {}

bool RelationalEvaluator::configure(FlexEmbeddings* tables, TokenLayout* layout,
                                    RelationalModel* model, std::string& err) {
  if (!tables || !layout || !model) {
    err = "null component";
    return false;
  }
  if (layout->tokens != model->config().tokens || layout->dim != model->config().dim) {
    err = "layout/model shape mismatch";
    return false;
  }
  tables_ = tables;
  layout_ = layout;
  model_ = model;
  acc_.configure(tables, layout);
  tokenBuf_.assign(layout->tokens * layout->dim, 0.0f);
  for (int i = 0; i < ContextSpec::kDim; ++i) ctx_[i] = 0.0f;
  return true;
}

void RelationalEvaluator::refresh(const Board& board) {
  std::vector<ActiveFeature> feats;
  GroupedFeatureSet::extract(board, feats);
  acc_.refresh(feats);
  computeContext(board, ctx_);
  phase_ = board.gamePhase();
}

void RelationalEvaluator::updateIncremental(const Board& board,
                                            const std::vector<ActiveFeature>& added,
                                            const std::vector<ActiveFeature>& removed) {
  acc_.applyDiff(added, removed);
  computeContext(board, ctx_);
  phase_ = board.gamePhase();
}

EvalResultFlex RelationalEvaluator::evaluate() const {
  acc_.tokens(tokenBuf_.data());
  EvalResultFlex r;
  model_->forwardWithContext(tokenBuf_.data(), ctx_, r.value, r.wdl, phase_);
  return r;
}

EvalResultFlex RelationalEvaluator::evaluateBoard(const Board& board) {
  refresh(board);
  return evaluate();
}

void RelationalEvaluator::currentTokens(float* out) const { acc_.tokens(out); }

void RelationalEvaluator::currentContext(float* out) const {
  for (int i = 0; i < ContextSpec::kDim; ++i) out[i] = ctx_[i];
}

}
