#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "core/incremental/relational_cache.h"

using namespace rune;
using Clock = std::chrono::high_resolution_clock;

static uint64_t g_state = 1307;

static float frand() {
  g_state = g_state * 6364136223846793005ULL + 1442695040888963407ULL;
  double u = static_cast<double>(g_state >> 33) / static_cast<double>(0xFFFFFFFFULL);
  return static_cast<float>((u - 0.5) * 0.16);
}

int main() {
  const int T = 8, D = 32;
  std::vector<float> wq(D * D), wk(D * D), wv(D * D), bq(D, 0.01f), bk(D, 0.01f),
      bv(D, 0.01f), gab(T * T, 0.05f);
  for (auto& x : wq) x = frand();
  for (auto& x : wk) x = frand();
  for (auto& x : wv) x = frand();
  v13::IncrWeights w;
  w.wq = wq.data();
  w.bq = bq.data();
  w.wk = wk.data();
  w.bk = bk.data();
  w.wv = wv.data();
  w.bv = bv.data();
  w.gab = gab.data();
  w.tokens = T;
  w.dim = D;
  v13::RelationalCache c;
  std::string err;
  if (!c.configure(w, 8, err)) {
    std::printf("configure failed: %s\n", err.c_str());
    return 1;
  }
  std::vector<float> x0(T * D, 0.5f), x1(T * D, 0.5f);
  x1[100] = 0.7f;
  x1[200] = 0.2f;
  const int reps = 20000;
  std::printf("changed,full_ns,incr_ns,cells\n");
  for (int k = 1; k <= 8; ++k) {
    std::vector<int> changed;
    for (int i = 0; i < k; ++i) changed.push_back(i);
    c.rebuild(x0.data(), nullptr);
    auto t0 = Clock::now();
    for (int r = 0; r < reps; ++r) c.rebuild(x1.data(), nullptr);
    auto t1 = Clock::now();
    c.rebuild(x0.data(), nullptr);
    auto t2 = Clock::now();
    for (int r = 0; r < reps; ++r) c.update(x1.data(), nullptr, changed);
    auto t3 = Clock::now();
    double fullNs =
        static_cast<double>((t1 - t0).count()) / reps;
    double incrNs =
        static_cast<double>((t3 - t2).count()) / reps;
    int cells = 2 * k * T - k * k;
    std::printf("%d,%.1f,%.1f,%d\n", k, fullNs, incrNs, cells);
  }
  auto bytes = c.cacheBytes();
  std::printf("bytes scores=%zu gates=%zu mixed=%zu\n", bytes.scores, bytes.gates,
              bytes.mixed);
  return 0;
}
