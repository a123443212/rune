#include "core/incremental/relational_cache.h"

#include <cmath>

#include "core/simd/simd.h"

namespace rune {
namespace v13 {

namespace {

float gateFn(bool hard, float x) {
  if (hard) {
    float y = 0.2f * x + 0.5f;
    if (y < 0.0f) y = 0.0f;
    if (y > 1.0f) y = 1.0f;
    return y;
  }
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

}

RelationalCache::RelationalCache() = default;

bool RelationalCache::configure(const IncrWeights& w, int threshold, std::string& err) {
  if (w.tokens <= 0 || w.dim <= 0) {
    err = "bad tokens/dim";
    return false;
  }
  if (!w.wq || !w.bq || !w.wk || !w.bk || !w.wv || !w.bv || !w.gab) {
    err = "null weights";
    return false;
  }
  if (w.ctxDim > 0 && (!w.dynU || !w.dynW)) {
    err = "dynamic bias needs dynU/dynW";
    return false;
  }
  w_ = w;
  threshold_ = threshold;
  size_t td = static_cast<size_t>(w.tokens) * static_cast<size_t>(w.dim);
  size_t tt = static_cast<size_t>(w.tokens) * static_cast<size_t>(w.tokens);
  x_.assign(td, 0.0f);
  q_.assign(td, 0.0f);
  k_.assign(td, 0.0f);
  v_.assign(td, 0.0f);
  s_.assign(tt, 0.0f);
  g_.assign(tt, 0.0f);
  y_.assign(td, 0.0f);
  out_.assign(td, 0.0f);
  u_.assign(static_cast<size_t>(w.tokens), 0.0f);
  wdyn_.assign(static_cast<size_t>(w.tokens), 0.0f);
  scratch_.assign(td, 0.0f);
  stack_.clear();
  fallbacks_ = 0;
  incr_ = 0;
  return true;
}

void RelationalCache::fullForward(const float* tokens, const float* ctx, float* q,
                                  float* k, float* vv, float* s, float* g, float* y,
                                  float* out) const {
  int t = w_.tokens;
  int d = w_.dim;
  for (int i = 0; i < t; ++i) {
    simd::matVec(w_.wq, tokens + i * d, w_.bq, q + i * d, d, d);
    simd::matVec(w_.wk, tokens + i * d, w_.bk, k + i * d, d, d);
    simd::matVec(w_.wv, tokens + i * d, w_.bv, vv + i * d, d, d);
  }
  simd::matMulTT(q, k, s, t, t, d);
  std::vector<float> uu(t, 0.0f), ww(t, 0.0f);
  bool dyn = ctx && w_.ctxDim > 0 && w_.dynU && w_.dynW;
  if (dyn) {
    for (int a = 0; a < t; ++a) {
      float su = 0.0f, sw = 0.0f;
      for (int c = 0; c < w_.ctxDim; ++c) {
        su += w_.dynU[a * w_.ctxDim + c] * ctx[c];
        sw += w_.dynW[a * w_.ctxDim + c] * ctx[c];
      }
      uu[a] = su;
      ww[a] = sw;
    }
  }
  for (int a = 0; a < t; ++a) {
    for (int b = 0; b < t; ++b) {
      float val = s[a * t + b] + w_.gab[a * t + b];
      if (dyn) {
        float dd = uu[a] * ww[b];
        if (dd < -0.25f) dd = -0.25f;
        if (dd > 0.25f) dd = 0.25f;
        val += dd;
      }
      s[a * t + b] = val;
      g[a * t + b] = gateFn(w_.hardGate, val);
    }
  }
  simd::matMul(g, vv, y, t, d, t);
  for (size_t i = 0; i < static_cast<size_t>(t) * static_cast<size_t>(d); ++i) {
    out[i] = tokens[i] + w_.alpha * y[i];
  }
}

void RelationalCache::rebuild(const float* tokens, const float* ctx) {
  size_t td = static_cast<size_t>(w_.tokens) * static_cast<size_t>(w_.dim);
  for (size_t i = 0; i < td; ++i) x_[i] = tokens[i];
  fullForward(tokens, ctx, q_.data(), k_.data(), v_.data(), s_.data(), g_.data(),
              y_.data(), out_.data());
  hadDyn_ = ctx && w_.ctxDim > 0 && w_.dynU && w_.dynW;
  if (ctx && w_.ctxDim > 0) {
    for (int a = 0; a < w_.tokens; ++a) {
      float su = 0.0f, sw = 0.0f;
      for (int c = 0; c < w_.ctxDim; ++c) {
        su += w_.dynU[a * w_.ctxDim + c] * ctx[c];
        sw += w_.dynW[a * w_.ctxDim + c] * ctx[c];
      }
      u_[a] = su;
      wdyn_[a] = sw;
    }
  }
  stack_.clear();
}

void RelationalCache::update(const float* tokensNew, const float* ctx,
                             const std::vector<int>& changed) {
  if (!useIncremental(changed)) {
    ++fallbacks_;
    rebuild(tokensNew, ctx);
    return;
  }
  ++incr_;
  int t = w_.tokens;
  int d = w_.dim;
  std::vector<int> mark(static_cast<size_t>(t), 0);
  for (int c : changed) {
    if (c >= 0 && c < t) mark[static_cast<size_t>(c)] = 1;
  }
  for (int c : changed) {
    if (c < 0 || c >= t) continue;
    simd::matVec(w_.wq, tokensNew + c * d, w_.bq, q_.data() + c * d, d, d);
    simd::matVec(w_.wk, tokensNew + c * d, w_.bk, k_.data() + c * d, d, d);
    simd::matVec(w_.wv, tokensNew + c * d, w_.bv, v_.data() + c * d, d, d);
  }
  bool dyn = ctx && w_.ctxDim > 0 && w_.dynU && w_.dynW;
  if (dyn) {
    for (int a = 0; a < t; ++a) {
      float su = 0.0f, sw = 0.0f;
      for (int c = 0; c < w_.ctxDim; ++c) {
        su += w_.dynU[a * w_.ctxDim + c] * ctx[c];
        sw += w_.dynW[a * w_.ctxDim + c] * ctx[c];
      }
      u_[a] = su;
      wdyn_[a] = sw;
    }
  }
  for (int a = 0; a < t; ++a) {
    for (int b = 0; b < t; ++b) {
      if (!dyn && !hadDyn_ && !mark[static_cast<size_t>(a)] && !mark[static_cast<size_t>(b)]) continue;
      float dot = 0.0f;
      for (int dd = 0; dd < d; ++dd) dot += q_[a * d + dd] * k_[b * d + dd];
      float val = dot + w_.gab[a * t + b];
      if (dyn) {
        float dd = u_[a] * wdyn_[b];
        if (dd < -0.25f) dd = -0.25f;
        if (dd > 0.25f) dd = 0.25f;
        val += dd;
      }
      s_[a * t + b] = val;
      g_[a * t + b] = gateFn(w_.hardGate, val);
    }
  }
  simd::matMul(g_.data(), v_.data(), y_.data(), t, d, t);
  size_t td = static_cast<size_t>(t) * static_cast<size_t>(d);
  for (size_t i = 0; i < td; ++i) x_[i] = tokensNew[i];
  for (size_t i = 0; i < td; ++i) out_[i] = x_[i] + w_.alpha * y_[i];
  hadDyn_ = dyn;
}

void RelationalCache::push() {
  Snapshot s;
  s.x = x_;
  s.q = q_;
  s.k = k_;
  s.v = v_;
  s.s = s_;
  s.g = g_;
  s.y = y_;
  s.out = out_;
  s.u = u_;
  s.wdyn = wdyn_;
  s.hadDyn = hadDyn_;
  stack_.push_back(std::move(s));
}

bool RelationalCache::pop() {
  if (stack_.empty()) return false;
  Snapshot s = std::move(stack_.back());
  stack_.pop_back();
  x_ = std::move(s.x);
  q_ = std::move(s.q);
  k_ = std::move(s.k);
  v_ = std::move(s.v);
  s_ = std::move(s.s);
  g_ = std::move(s.g);
  y_ = std::move(s.y);
  out_ = std::move(s.out);
  u_ = std::move(s.u);
  wdyn_ = std::move(s.wdyn);
  hadDyn_ = s.hadDyn;
  return true;
}

bool RelationalCache::verifyAgainstFull(const float* tokensNew, const float* ctx,
                                       float tol, float* maxDiff) const {
  std::vector<float> q(q_.size()), k(k_.size()), vv(v_.size()), s(s_.size()),
      g(g_.size()), y(y_.size()), out(out_.size());
  fullForward(tokensNew, ctx, q.data(), k.data(), vv.data(), s.data(), g.data(),
              y.data(), out.data());
  const std::vector<const std::vector<float>*> a = {&q_, &k_, &v_, &s_, &g_, &y_, &out_};
  const std::vector<const std::vector<float>*> b = {&q, &k, &vv, &s, &g, &y, &out};
  float worst = 0.0f;
  for (size_t p = 0; p < a.size(); ++p) {
    for (size_t i = 0; i < a[p]->size(); ++i) {
      float dd = std::fabs((*a[p])[i] - (*b[p])[i]);
      if (dd > worst) worst = dd;
    }
  }
  if (maxDiff) *maxDiff = worst;
  return worst <= tol;
}

CacheBytes RelationalCache::cacheBytes() const {
  CacheBytes b;
  b.scores = s_.size() * sizeof(float);
  b.gates = g_.size() * sizeof(float);
  b.mixed = y_.size() * sizeof(float);
  b.scratch = scratch_.size() * sizeof(float);
  return b;
}

}
}
