#include "core/architectures/adaptive/adaptive.h"

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

bool adaptiveModeFromString(const std::string& name, AdaptiveMode& out) {
  if (name == "cheap") {
    out = AdaptiveMode::Cheap;
    return true;
  }
  if (name == "always") {
    out = AdaptiveMode::Always;
    return true;
  }
  if (name == "adaptive") {
    out = AdaptiveMode::Adaptive;
    return true;
  }
  return false;
}

const char* adaptiveModeName(AdaptiveMode mode) {
  if (mode == AdaptiveMode::Cheap) return "cheap";
  if (mode == AdaptiveMode::Always) return "always";
  return "adaptive";
}

bool searchRouteFromString(const std::string& name, SearchRoute& out) {
  if (name == "difficulty") {
    out = SearchRoute::Difficulty;
    return true;
  }
  if (name == "uncertainty") {
    out = SearchRoute::Uncertainty;
    return true;
  }
  if (name == "both") {
    out = SearchRoute::Both;
    return true;
  }
  if (name == "full") {
    out = SearchRoute::Full;
    return true;
  }
  return false;
}

const char* searchRouteName(SearchRoute route) {
  if (route == SearchRoute::Difficulty) return "difficulty";
  if (route == SearchRoute::Uncertainty) return "uncertainty";
  if (route == SearchRoute::Both) return "both";
  return "full";
}

float AdaptiveModel::clip01(float v) {
  if (v < 0.0f) return 0.0f;
  if (v > 1.0f) return 1.0f;
  return v;
}

AdaptiveModel::AdaptiveModel() {
  AdaptiveBuildSpec spec;
  std::string err;
  configure(spec, err);
}

bool AdaptiveModel::configure(const AdaptiveBuildSpec& spec, std::string& err) {
  if (spec.dim < 8 || spec.dim > 64) {
    err = "bad adaptive dim";
    return false;
  }
  PoolMode mode;
  if (!poolModeFromString(spec.cheapPooling, mode)) {
    err = "unknown cheap pooling";
    return false;
  }
  if (mode != PoolMode::None && mode != PoolMode::Shared) {
    err = "cheap pooling must be none or shared";
    return false;
  }
  if (mode == PoolMode::Shared && spec.dim != 32) {
    err = "shared cheap pooling requires dim 32";
    return false;
  }
  for (const auto& p : spec.prunedPairs) {
    if (p.first < 0 || p.first > 7 || p.second < 0 || p.second > 7) {
      err = "pruned pair out of range";
      return false;
    }
  }
  bspec_ = spec;
  dim_ = spec.dim;
  total_ = 8 * dim_;
  if (spec.cheapHidden < 4 || spec.cheapHidden > 128 || spec.refH1 < 4 ||
      spec.refH1 > 512 || spec.refH2 < 4 || spec.refH2 > 512) {
    err = "bad adaptive head widths";
    return false;
  }
  cheapHidden_ = spec.cheapHidden;
  refH1_ = spec.refH1;
  refH2_ = spec.refH2;
  if (spec.hasUncertainty) {
    archId_ = "RUNE-05";
    archVersion_ = "0.5.0";
  } else {
    archId_ = "RUNE-04";
    archVersion_ = "0.4.0";
  }
  VarWidths widths;
  for (int g = 0; g < 8; ++g) widths.w[g] = dim_;
  if (!pool.configure(widths, mode, true, 32, err)) return false;
  int in = total_;
  int ch = cheapHidden_;
  int r1 = refH1_;
  int r2 = refH2_;
  uint64_t s = 40400 + dim_;
  initVec(cw1, ch * in, s, 0.05f);
  initVec(cb1, ch, s, 0.01f);
  initVec(cwv, 1 * ch, s, 0.05f);
  initVec(cbv, 1, s, 0.01f);
  initVec(cww, 3 * ch, s, 0.05f);
  initVec(cbw, 3, s, 0.01f);
  initVec(dw, 1 * in, s, 0.01f);
  initVec(db, 1, s, 0.0f);
  initVec(uw, 1 * in, s, 0.01f);
  initVec(ub, 1, s, 0.0f);
  initVec(sw, 1 * in, s, 0.01f);
  initVec(sb, 1, s, 0.0f);
  initVec(wq, dim_ * dim_, s, 0.08f);
  initVec(bq, dim_, s, 0.01f);
  initVec(wk, dim_ * dim_, s, 0.08f);
  initVec(bk, dim_, s, 0.01f);
  initVec(wv, dim_ * dim_, s, 0.08f);
  initVec(bv, dim_, s, 0.01f);
  gabS.assign(64, 0.0f);
  initVec(w1, r1 * in, s, 0.05f);
  initVec(b1, r1, s, 0.01f);
  initVec(w2, r2 * r1, s, 0.05f);
  initVec(b2, r2, s, 0.01f);
  initVec(wvo, 1 * r2, s, 0.05f);
  initVec(bvo, 1, s, 0.01f);
  initVec(wwdl, 3 * r2, s, 0.05f);
  initVec(bwdl, 3, s, 0.01f);
  for (int i = 0; i < 64; ++i) pruneMask_[i] = true;
  for (const auto& p : spec.prunedPairs) pruneMask_[p.first * 8 + p.second] = false;
  scratch_.assign(ch + 3 * in + 64 + 64 + in + r1 + r2, 0.0f);
  return true;
}

void AdaptiveModel::cheapForward(const float* acc, float* cheapFlat, float& value, float* wdl,
                                 float& difficulty) const {
  int ch = cheapHidden_;
  pool.forward(acc, cheapFlat);
  float* h = scratch_.data();
  simd::matVecClipped(cw1.data(), cheapFlat, cb1.data(), h, ch, total_);
  float vv = cbv[0];
  for (int i = 0; i < ch; ++i) vv += cwv[i] * h[i];
  value = std::tanh(vv);
  simd::matVec(cww.data(), h, cbw.data(), wdl, 3, ch);
  float d = db[0];
  for (int i = 0; i < total_; ++i) d += dw[i] * cheapFlat[i];
  difficulty = d;
}

void AdaptiveModel::refineForward(const float* cheapFlat, float& value, float* wdl) const {
  int d = dim_;
  int ch = cheapHidden_;
  int r1 = refH1_;
  int r2 = refH2_;
  float* q = scratch_.data() + ch;
  float* k = q + total_;
  float* v = k + total_;
  float* s = v + total_;
  float* y = s + 64;
  float* mixed = y + 64;
  float* h1 = mixed + total_;
  float* h2 = h1 + r1;
  for (int i = 0; i < 8; ++i) {
    simd::matVec(wq.data(), cheapFlat + i * d, bq.data(), q + i * d, d, d);
    simd::matVec(wk.data(), cheapFlat + i * d, bk.data(), k + i * d, d, d);
    simd::matVec(wv.data(), cheapFlat + i * d, bv.data(), v + i * d, d, d);
  }
  simd::matMulTT(q, k, s, 8, 8, d);
  for (int i = 0; i < 64; ++i) {
    float g = clip01(s[i] + gabS[i]);
    s[i] = pruneMask_[i] ? g : 0.0f;
  }
  simd::matMul(s, v, y, 8, d, 8);
  for (int i = 0; i < total_; ++i) mixed[i] = cheapFlat[i] + bspec_.alpha * y[i];
  simd::matVecClipped(w1.data(), mixed, b1.data(), h1, r1, total_);
  simd::matVecClipped(w2.data(), h1, b2.data(), h2, r2, r1);
  float vv = bvo[0];
  for (int i = 0; i < r2; ++i) vv += wvo[i] * h2[i];
  value = std::tanh(vv);
  simd::matVec(wwdl.data(), h2, bwdl.data(), wdl, 3, r2);
}

bool AdaptiveModel::route(float difficulty, bool prev, float threshold) const {
  if (!(difficulty == difficulty)) {
    return true;
  }
  if (bspec_.hasTLow) {
    if (prev) [[likely]] {
      return difficulty >= bspec_.tLow;
    } else {
      return difficulty >= threshold;
    }
  }
  return difficulty >= threshold;
}

bool AdaptiveModel::route(float difficulty, bool prev) const {
  return route(difficulty, prev, bspec_.threshold);
}

float AdaptiveModel::uncertaintyForward(const float* cheapFlat) const {
  float z = ub[0];
  for (int i = 0; i < total_; ++i) z += uw[i] * cheapFlat[i];
  return 1.0f / (1.0f + std::exp(-z));
}

float AdaptiveModel::stabilityForward(const float* cheapFlat) const {
  float v = sb[0];
  for (int i = 0; i < total_; ++i) v += sw[i] * cheapFlat[i];
  return v < 0.0f ? 0.0f : v;
}

bool AdaptiveModel::routeSearch(SearchRoute route, float difficulty, float uncertainty,
                                float stability, bool prev, const RoutingThresholds& t) const {
  bool needDiff;
  if (!(difficulty == difficulty)) {
    needDiff = true;
  } else if (difficulty >= t.tHigh) {
    needDiff = true;
  } else if (t.hasTLow) {
    needDiff = prev ? difficulty >= t.tLow : difficulty >= t.diffT;
  } else {
    needDiff = difficulty >= t.diffT;
  }
  bool needUnc = uncertainty >= t.uncT;
  if (route == SearchRoute::Difficulty) return needDiff;
  if (route == SearchRoute::Uncertainty) return needUnc;
  if (route == SearchRoute::Both) return needDiff || needUnc;
  return needDiff || needUnc || (stability >= t.stabT);
}

void AdaptiveModel::forward(const float* tokens, float& value, float* wdl, int phase) const {
  (void)phase;
  std::vector<float> cheap(total_);
  float diff = 0.0f;
  float cv = 0.0f;
  float cw[3] = {0.0f, 0.0f, 0.0f};
  cheapForward(tokens, cheap.data(), cv, cw, diff);
  refineForward(cheap.data(), value, wdl);
}

size_t AdaptiveModel::parameterCount() const {
  size_t n = pool.parameterCount() + cw1.size() + cb1.size() + cwv.size() + cbv.size() +
             cww.size() + cbw.size() + dw.size() + db.size() + wq.size() + bq.size() +
             wk.size() + bk.size() + wv.size() + bv.size() + gabS.size() + w1.size() +
             b1.size() + w2.size() + b2.size() + wvo.size() + bvo.size() + wwdl.size() +
             bwdl.size();
  if (bspec_.hasUncertainty) n += uw.size() + ub.size() + sw.size() + sb.size();
  return n;
}

size_t AdaptiveModel::cheapParameterCount() const {
  size_t n = pool.parameterCount() + cw1.size() + cb1.size() + cwv.size() + cbv.size() +
             cww.size() + cbw.size() + dw.size() + db.size();
  if (bspec_.hasUncertainty) n += uw.size() + ub.size();
  return n;
}

void AdaptiveModel::getTensors(std::vector<std::string>& names,
                               std::vector<std::vector<int>>& shapes,
                               std::vector<const float*>& data) const {
  names.clear();
  shapes.clear();
  data.clear();
  if (pool.mode() == PoolMode::Shared) {
    names.push_back("pool_S");
    shapes.push_back({32, 32});
    data.push_back(pool.sharedS.data());
    for (int t = 0; t < 8; ++t) {
      int w = dim_;
      names.push_back("pool_s" + std::to_string(t));
      names.push_back("pool_b" + std::to_string(t));
      shapes.push_back({w});
      shapes.push_back({w});
      data.push_back(pool.tokS[t].data());
      data.push_back(pool.tokB[t].data());
    }
  }
  int in = total_;
  int d = dim_;
  int ch = cheapHidden_;
  int r1 = refH1_;
  int r2 = refH2_;
  const std::vector<std::string> tail = {"cw1", "cb1", "cwv", "cbv", "cww", "cbw",
                                         "dw",    "db",  "wq",  "bq",  "wk",  "bk",
                                         "wvv",   "bvv", "gabS", "w1", "b1",  "w2",
                                         "b2",    "wvo", "bvo", "wwdl", "bwdl"};
  const std::vector<std::vector<int>> tailShapes = {
      {ch, in}, {ch}, {1, ch}, {1}, {3, ch}, {3}, {1, in}, {1}, {d, d}, {d},
      {d, d},   {d},  {d, d},  {d}, {8, 8},   {r1, in},   {r1},  {r2, r1},
      {r2},     {1, r2}, {1},  {3, r2},       {3}};
  for (const auto& n : tail) names.push_back(n);
  for (const auto& sh : tailShapes) shapes.push_back(sh);
  if (bspec_.hasUncertainty) {
    const std::vector<std::string> aux = {"uw", "ub", "sw", "sb"};
    const std::vector<std::vector<int>> auxShapes = {{1, in}, {1}, {1, in}, {1}};
    for (const auto& n : aux) names.push_back(n);
    for (const auto& sh : auxShapes) shapes.push_back(sh);
  }
  data.insert(data.end(), {cw1.data(), cb1.data(), cwv.data(), cbv.data(), cww.data(),
                           cbw.data(), dw.data(), db.data(), wq.data(), bq.data(),
                           wk.data(), bk.data(), wv.data(), bv.data(), gabS.data(),
                           w1.data(), b1.data(), w2.data(), b2.data(), wvo.data(),
                           bvo.data(), wwdl.data(), bwdl.data()});
  if (bspec_.hasUncertainty) {
    data.insert(data.end(), {uw.data(), ub.data(), sw.data(), sb.data()});
  }
}

bool AdaptiveModel::setTensors(const std::vector<std::string>& names,
                              const std::vector<float>& flat) {
  std::vector<std::string> want;
  std::vector<std::vector<float>*> slots;
  if (pool.mode() == PoolMode::Shared) {
    want.push_back("pool_S");
    slots.push_back(&pool.sharedS);
    for (int t = 0; t < 8; ++t) {
      want.push_back("pool_s" + std::to_string(t));
      want.push_back("pool_b" + std::to_string(t));
      slots.push_back(&pool.tokS[t]);
      slots.push_back(&pool.tokB[t]);
    }
  }
  for (const char* n :
       {"cw1", "cb1", "cwv", "cbv", "cww", "cbw", "dw", "db", "wq", "bq", "wk", "bk",
        "wvv", "bvv", "gabS", "w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"})
    want.push_back(n);
  slots.insert(slots.end(), {&cw1, &cb1, &cwv, &cbv, &cww, &cbw, &dw, &db, &wq, &bq,
                             &wk, &bk, &wv, &bv, &gabS, &w1, &b1, &w2, &b2, &wvo,
                             &bvo, &wwdl, &bwdl});
  if (bspec_.hasUncertainty) {
    for (const char* n : {"uw", "ub", "sw", "sb"}) want.push_back(n);
    slots.insert(slots.end(), {&uw, &ub, &sw, &sb});
  }
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

ModelSpec AdaptiveModel::spec() const {
  ModelSpec s;
  s.arch = archId();
  s.archVersion = archVersion();
  s.tokens = 8;
  s.tokenDim = dim_;
  s.tokenDims.clear();
  for (int t = 0; t < 8; ++t) s.tokenDims.push_back(dim_);
  s.attention = "static_refinement";
  s.geometricBias = "static";
  s.head = "cheap_plus_refined";
  s.cheapPooling = poolModeName(pool.mode());
  s.threshold = bspec_.threshold;
  s.tHigh = bspec_.tHigh;
  s.hasTLow = bspec_.hasTLow;
  s.tLow = bspec_.tLow;
  s.refinePrecision = bspec_.refinePrecision;
  s.prunedPairs = bspec_.prunedPairs;
  s.hasUncertainty = bspec_.hasUncertainty;
  s.hasStabilityHead = bspec_.hasStabilityHead;
  s.cheapHidden = cheapHidden_;
  s.refH1 = refH1_;
  s.refH2 = refH2_;
  return s;
}

AdaptiveEvaluator::AdaptiveEvaluator() {}

bool AdaptiveEvaluator::configure(VarEmbeddings* tables, AdaptiveModel* model, std::string& err) {
  if (!tables || !model) {
    err = "null component";
    return false;
  }
  tables_ = tables;
  model_ = model;
  acc_.configure(tables);
  int total = 0;
  for (int g = 0; g < 8; ++g) total += tables->groupWidth(g);
  tokenBuf_.assign(total, 0.0f);
  cheapBuf_.assign(total, 0.0f);
  return true;
}

void AdaptiveEvaluator::refresh(const Board& board) {
  std::vector<ActiveFeature> feats;
  GroupedFeatureSet::extract(board, feats);
  acc_.refresh(feats);
}

void AdaptiveEvaluator::updateIncremental(const std::vector<ActiveFeature>& added,
                                          const std::vector<ActiveFeature>& removed) {
  acc_.applyDiff(added, removed);
}

AdaptiveEvalResult AdaptiveEvaluator::evaluate(AdaptiveMode mode, float thresholdOverride,
                                               bool useOverride, bool prev) const {
  AdaptiveEvalResult r;
  acc_.tokens(tokenBuf_.data());
  float diff = 0.0f;
  model_->cheapForward(tokenBuf_.data(), cheapBuf_.data(), r.value, r.wdl, diff);
  r.difficulty = diff;
  float threshold = useOverride ? thresholdOverride : model_->buildSpec().threshold;
  bool refine = false;
  if (mode == AdaptiveMode::Always) refine = true;
  else if (mode == AdaptiveMode::Adaptive) refine = model_->route(diff, prev, threshold);
  r.refined = refine;
  if (refine) model_->refineForward(cheapBuf_.data(), r.value, r.wdl);
  return r;
}

AdaptiveEvalResult AdaptiveEvaluator::evaluateBoard(const Board& board, AdaptiveMode mode,
                                                    float thresholdOverride, bool useOverride,
                                                    bool prev) {
  refresh(board);
  return evaluate(mode, thresholdOverride, useOverride, prev);
}

void AdaptiveEvaluator::currentTokens(float* out) const { acc_.tokens(out); }

void AdaptiveEvaluator::currentCheap(float* out) const {
  acc_.tokens(tokenBuf_.data());
  float v = 0.0f;
  float w[3] = {0.0f, 0.0f, 0.0f};
  float d = 0.0f;
  model_->cheapForward(tokenBuf_.data(), out, v, w, d);
}

AdaptiveEvalResult AdaptiveEvaluator::evaluateSearch(const Board& board, SearchRoute route,
                                                     const RoutingThresholds& t, bool prev) {
  AdaptiveEvalResult r;
  refresh(board);
  acc_.tokens(tokenBuf_.data());
  float diff = 0.0f;
  model_->cheapForward(tokenBuf_.data(), cheapBuf_.data(), r.value, r.wdl, diff);
  r.difficulty = diff;
  r.uncertainty = model_->uncertaintyForward(cheapBuf_.data());
  r.stability = model_->stabilityForward(cheapBuf_.data());
  r.refined = model_->routeSearch(route, diff, r.uncertainty, r.stability, prev, t);
  if (r.refined) model_->refineForward(cheapBuf_.data(), r.value, r.wdl);
  return r;
}

}
