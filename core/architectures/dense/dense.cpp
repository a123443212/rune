#include "core/architectures/dense/dense.h"

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

DenseModel::DenseModel() {
  DenseBuildSpec spec;
  std::string err;
  configure(spec, err);
}

bool DenseModel::configure(const DenseBuildSpec& spec, std::string& err) {
  if (spec.variant != "A" && spec.variant != "B" && spec.variant != "C" && spec.variant != "D") {
    err = "unknown variant";
    return false;
  }
  VarWidths widths;
  if (!VarWidths::make(spec.dims, widths, err)) return false;
  PoolMode mode;
  if (!poolModeFromString(spec.pooling, mode)) {
    err = "unknown pooling";
    return false;
  }
  if (mode == PoolMode::Shared) {
    for (int d : spec.dims) {
      if (d > spec.sharedWidth) {
        err = "token dim exceeds shared width";
        return false;
      }
    }
  }
  if (spec.headH1 < 4 || spec.headH1 > 512 || spec.headH2 < 4 || spec.headH2 > 512) {
    err = "bad head widths";
    return false;
  }
  bspec_ = spec;
  widths_ = widths;
  headH1_ = spec.headH1;
  headH2_ = spec.headH2;
  archId_ = "RUNE-03-" + spec.variant;
  if (!pool.configure(widths, mode, spec.poolClip, spec.sharedWidth, err)) return false;
  if (!gate.configure(widths, spec.gateOn, err)) return false;
  int in = widths.total();
  int h1 = headH1_;
  int h2 = headH2_;
  uint64_t s = 91030 + widths.total();
  initVec(w1, h1 * in, s, 0.05f);
  initVec(b1, h1, s, 0.01f);
  initVec(w2, h2 * h1, s, 0.05f);
  initVec(b2, h2, s, 0.01f);
  initVec(wvo, 1 * h2, s, 0.05f);
  initVec(bvo, 1, s, 0.01f);
  initVec(wwdl, 3 * h2, s, 0.05f);
  initVec(bwdl, 3, s, 0.01f);
  scratch_.assign(2 * in + h1 + h2, 0.0f);
  return true;
}

void DenseModel::forward(const float* tokens, float& value, float* wdl, int phase) const {
  (void)phase;
  int in = widths_.total();
  int h1 = headH1_;
  int h2 = headH2_;
  float* pooled = scratch_.data();
  float* gated = scratch_.data() + in;
  float* h1buf = scratch_.data() + 2 * in;
  float* h2buf = scratch_.data() + 2 * in + h1;
  pool.forward(tokens, pooled);
  gate.forward(pooled, gated);
  simd::matVecClipped(w1.data(), gated, b1.data(), h1buf, h1, in);
  simd::matVecClipped(w2.data(), h1buf, b2.data(), h2buf, h2, h1);
  float vv = bvo[0];
  for (int i = 0; i < h2; ++i) vv += wvo[i] * h2buf[i];
  value = std::tanh(vv);
  simd::matVec(wwdl.data(), h2buf, bwdl.data(), wdl, 3, h2);
}

size_t DenseModel::parameterCount() const {
  return pool.parameterCount() + gate.parameterCount() + w1.size() + b1.size() + w2.size() +
         b2.size() + wvo.size() + bvo.size() + wwdl.size() + bwdl.size();
}

void DenseModel::getTensors(std::vector<std::string>& names,
                            std::vector<std::vector<int>>& shapes,
                            std::vector<const float*>& data) const {
  names.clear();
  shapes.clear();
  data.clear();
  PoolMode mode = pool.mode();
  if (mode == PoolMode::PerToken) {
    for (int t = 0; t < 8; ++t) {
      int w = widths_.w[t];
      names.push_back("pool_w" + std::to_string(t));
      names.push_back("pool_b" + std::to_string(t));
      shapes.push_back({w, w});
      shapes.push_back({w});
      data.push_back(pool.perW[t].data());
      data.push_back(pool.perB[t].data());
    }
  } else if (mode == PoolMode::Shared) {
    int sw = bspec_.sharedWidth;
    names.push_back("pool_S");
    shapes.push_back({sw, sw});
    data.push_back(pool.sharedS.data());
    for (int t = 0; t < 8; ++t) {
      int w = widths_.w[t];
      names.push_back("pool_s" + std::to_string(t));
      names.push_back("pool_b" + std::to_string(t));
      shapes.push_back({w});
      shapes.push_back({w});
      data.push_back(pool.tokS[t].data());
      data.push_back(pool.tokB[t].data());
    }
  }
  if (gate.enabled()) {
    names.push_back("gate_a");
    names.push_back("gate_b");
    shapes.push_back({widths_.total()});
    shapes.push_back({widths_.total()});
    data.push_back(gate.ga.data());
    data.push_back(gate.gb.data());
  }
  int in = widths_.total();
  int h1 = headH1_;
  int h2 = headH2_;
  names.insert(names.end(), {"w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"});
  shapes.insert(shapes.end(), {{h1, in}, {h1}, {h2, h1}, {h2}, {1, h2}, {1}, {3, h2}, {3}});
  data.insert(data.end(), {w1.data(), b1.data(), w2.data(), b2.data(), wvo.data(), bvo.data(),
                           wwdl.data(), bwdl.data()});
}

bool DenseModel::setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) {
  std::vector<std::string> want;
  std::vector<std::vector<float>*> slots;
  PoolMode mode = pool.mode();
  if (mode == PoolMode::PerToken) {
    for (int t = 0; t < 8; ++t) {
      want.push_back("pool_w" + std::to_string(t));
      want.push_back("pool_b" + std::to_string(t));
      slots.push_back(&pool.perW[t]);
      slots.push_back(&pool.perB[t]);
    }
  } else if (mode == PoolMode::Shared) {
    want.push_back("pool_S");
    slots.push_back(&pool.sharedS);
    for (int t = 0; t < 8; ++t) {
      want.push_back("pool_s" + std::to_string(t));
      want.push_back("pool_b" + std::to_string(t));
      slots.push_back(&pool.tokS[t]);
      slots.push_back(&pool.tokB[t]);
    }
  }
  if (gate.enabled()) {
    want.push_back("gate_a");
    want.push_back("gate_b");
    slots.push_back(&gate.ga);
    slots.push_back(&gate.gb);
  }
  for (const char* n : {"w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"}) want.push_back(n);
  slots.insert(slots.end(), {&w1, &b1, &w2, &b2, &wvo, &bvo, &wwdl, &bwdl});
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

ModelSpec DenseModel::spec() const {
  ModelSpec s;
  s.arch = archId_;
  s.archVersion = archVersion();
  s.tokens = 8;
  s.tokenDims = {widths_.w[0], widths_.w[1], widths_.w[2], widths_.w[3],
                 widths_.w[4], widths_.w[5], widths_.w[6], widths_.w[7]};
  s.tokenDim = widths_.total();
  s.attention = "none";
  s.geometricBias = "none";
  s.pooling = poolModeName(pool.mode());
  s.gateOn = gate.enabled();
  s.poolClip = pool.clipOut();
  s.variant = bspec_.variant;
  s.headH1 = headH1_;
  s.headH2 = headH2_;
  return s;
}

DenseEvaluator::DenseEvaluator() {}

bool DenseEvaluator::configure(VarEmbeddings* tables, DenseModel* model, std::string& err) {
  if (!tables || !model) {
    err = "null component";
    return false;
  }
  tables_ = tables;
  model_ = model;
  acc_.configure(tables);
  if (model->pool.mode() == PoolMode::Shared) {
    for (int g = 0; g < 8; ++g) {
      if (tables->groupWidth(g) != model->buildSpec().sharedWidth) {
        err = "shared pooling needs uniform group width";
        return false;
      }
    }
  }
  tokenBuf_.assign(512, 0.0f);
  return true;
}

void DenseEvaluator::refresh(const Board& board) {
  std::vector<ActiveFeature> feats;
  GroupedFeatureSet::extract(board, feats);
  acc_.refresh(feats);
}

void DenseEvaluator::updateIncremental(const std::vector<ActiveFeature>& added,
                                       const std::vector<ActiveFeature>& removed) {
  acc_.applyDiff(added, removed);
}

void DenseEvaluator::evaluate(float& value, float* wdl) const {
  acc_.tokens(tokenBuf_.data());
  model_->forward(tokenBuf_.data(), value, wdl);
}

void DenseEvaluator::evaluateBoard(const Board& board, float& value, float* wdl) {
  refresh(board);
  evaluate(value, wdl);
}

void DenseEvaluator::currentTokens(float* out) const { acc_.tokens(out); }

}
