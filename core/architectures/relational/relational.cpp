#include "core/architectures/relational/relational.h"

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

float applyGate(GateFn fn, float s) {
  if (fn == GateFn::HardSigmoid) {
    float v = 0.2f * s + 0.5f;
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
  }
  if (s < 0.0f) return 0.0f;
  if (s > 1.0f) return 1.0f;
  return s;
}

}  // namespace

bool gateFromString(const std::string& name, GateFn& out) {
  if (name == "clip") {
    out = GateFn::Clip;
    return true;
  }
  if (name == "hard_sigmoid") {
    out = GateFn::HardSigmoid;
    return true;
  }
  return false;
}

const char* gateName(GateFn fn) { return fn == GateFn::HardSigmoid ? "hard_sigmoid" : "clip"; }

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
  initVec(w1, 128 * in, s, 0.05f);
  initVec(b1, 128, s, 0.01f);
  initVec(w2, 32 * 128, s, 0.05f);
  initVec(b2, 32, s, 0.01f);
  initVec(wvo, 1 * 32, s, 0.05f);
  initVec(bvo, 1, s, 0.01f);
  initVec(wwdl, 3 * 32, s, 0.05f);
  initVec(bwdl, 3, s, 0.01f);
  scratch_.assign(in + 128 + 32, 0.0f);
  return true;
}

void RelationalModel::forward(const float* tokens, float& value, float* wdl) const {
  float ctx[ContextSpec::kDim] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  forwardWithContext(tokens, ctx, value, wdl);
}

void RelationalModel::forwardWithContext(const float* tokens, const float* ctx, float& value,
                                         float* wdl) const {
  int in = mixer.config().tokens * mixer.config().dim;
  float* mixed = scratch_.data();
  float* h1 = scratch_.data() + in;
  float* h2 = scratch_.data() + in + 128;
  mixer.forward(tokens, ctx, mixed);
  simd::matVecClipped(w1.data(), mixed, b1.data(), h1, 128, in);
  simd::matVecClipped(w2.data(), h1, b2.data(), h2, 32, 128);
  float vv = bvo[0];
  for (int i = 0; i < 32; ++i) vv += wvo[i] * h2[i];
  value = std::tanh(vv);
  simd::matVec(wwdl.data(), h2, bwdl.data(), wdl, 3, 32);
}

size_t RelationalModel::parameterCount() const {
  return mixer.parameterCount() + w1.size() + b1.size() + w2.size() + b2.size() + wvo.size() +
         bvo.size() + wwdl.size() + bwdl.size();
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
  names.insert(names.end(), {"w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"});
  shapes.insert(shapes.end(), {{128, in}, {128}, {32, 128}, {32}, {1, 32}, {1}, {3, 32}, {3}});
  data.insert(data.end(), {w1.data(), b1.data(), w2.data(), b2.data(), wvo.data(), bvo.data(),
                           wwdl.data(), bwdl.data()});
}

bool RelationalModel::setTensors(const std::vector<std::string>& names,
                                 const std::vector<float>& flat) {
  std::vector<std::string> want = {"wq", "bq", "wk", "bk", "wvv", "bvv", "gabS"};
  if (mixer.config().dynamicBias) {
    want.push_back("dynU");
    want.push_back("dynW");
  }
  for (const char* n : {"w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"}) want.push_back(n);
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
  slots.insert(slots.end(), {&w1, &b1, &w2, &b2, &wvo, &bvo, &wwdl, &bwdl});
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
  s.archVersion = archVersion();
  s.tokens = mixer.config().tokens;
  s.tokenDim = mixer.config().dim;
  s.attention = "gated_relational";
  s.geometricBias = mixer.config().dynamicBias ? "dynamic" : "static";
  s.gate = gateName(mixer.config().gate);
  s.alpha = mixer.config().alpha;
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
}

void RelationalEvaluator::updateIncremental(const Board& board,
                                            const std::vector<ActiveFeature>& added,
                                            const std::vector<ActiveFeature>& removed) {
  acc_.applyDiff(added, removed);
  computeContext(board, ctx_);
}

EvalResultFlex RelationalEvaluator::evaluate() const {
  acc_.tokens(tokenBuf_.data());
  EvalResultFlex r;
  model_->forwardWithContext(tokenBuf_.data(), ctx_, r.value, r.wdl);
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
