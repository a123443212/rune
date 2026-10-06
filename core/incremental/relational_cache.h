#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace rune {
namespace v13 {

struct IncrWeights {
  const float* wq = nullptr;
  const float* bq = nullptr;
  const float* wk = nullptr;
  const float* bk = nullptr;
  const float* wv = nullptr;
  const float* bv = nullptr;
  const float* gab = nullptr;
  const float* dynU = nullptr;
  const float* dynW = nullptr;
  int tokens = 8;
  int dim = 32;
  int ctxDim = 0;
  bool hardGate = false;
  float alpha = 1.0f;
};

struct CacheBytes {
  size_t scores = 0;
  size_t gates = 0;
  size_t mixed = 0;
  size_t scratch = 0;
};

class RelationalCache {
 public:
  RelationalCache();

  bool configure(const IncrWeights& w, int threshold, std::string& err);
  void rebuild(const float* tokens, const float* ctx);
  void update(const float* tokensNew, const float* ctx, const std::vector<int>& changed);

  bool useIncremental(const std::vector<int>& changed) const { return static_cast<int>(changed.size()) <= threshold_; }

  void push();
  bool pop();

  bool verifyAgainstFull(const float* tokensNew, const float* ctx, float tol,
                         float* maxDiff) const;

  const std::vector<float>& q() const { return q_; }
  const std::vector<float>& k() const { return k_; }
  const std::vector<float>& v() const { return v_; }
  const std::vector<float>& scores() const { return s_; }
  const std::vector<float>& gates() const { return g_; }
  const std::vector<float>& mixed() const { return y_; }
  const std::vector<float>& out() const { return out_; }
  const std::vector<float>& tokens() const { return x_; }

  int fallbacks() const { return fallbacks_; }
  int incrementalUpdates() const { return incr_; }
  int threshold() const { return threshold_; }
  void setThreshold(int t) { threshold_ = t; }
  CacheBytes cacheBytes() const;

 private:
  void fullForward(const float* tokens, const float* ctx, float* q, float* k, float* vv,
                   float* s, float* g, float* y, float* out) const;

  IncrWeights w_;
  int threshold_ = 2;
  std::vector<float> x_, q_, k_, v_, s_, g_, y_, out_, u_, wdyn_;
  bool hadDyn_ = false;
  struct Snapshot {
    std::vector<float> x, q, k, v, s, g, y, out, u, wdyn;
    bool hadDyn = false;
  };
  std::vector<Snapshot> stack_;
  int fallbacks_ = 0;
  int incr_ = 0;
  mutable std::vector<float> scratch_;
};

}
}
