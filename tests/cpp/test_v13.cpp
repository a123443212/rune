#include "tests/cpp/test_v13.h"
#include "tests/cpp/test_framework.h"
#include <cmath>
#include <vector>
#include "core/incremental/token_delta.h"
#include "core/incremental/interaction_graph.h"
#include "core/incremental/relational_cache.h"
#include "core/incremental/incremental_mixer.h"
#include "core/architectures/attention/attention.h"
#include <fstream>
#include <sstream>
using namespace rune;

static void fillWeights(int t, int d, std::vector<float>& wq, std::vector<float>& bq,
                        std::vector<float>& wk, std::vector<float>& bk,
                        std::vector<float>& wv, std::vector<float>& bv,
                        std::vector<float>& gab) {
  uint64_t s = 777;
  auto rnd = [&]() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    double u = static_cast<double>(s >> 33) / static_cast<double>(0xFFFFFFFFULL);
    return static_cast<float>((u - 0.5) * 0.16);
  };
  wq.assign(static_cast<size_t>(d * d), 0);
  wk.assign(static_cast<size_t>(d * d), 0);
  wv.assign(static_cast<size_t>(d * d), 0);
  for (auto& x : wq) x = rnd();
  for (auto& x : wk) x = rnd();
  for (auto& x : wv) x = rnd();
  bq.assign(d, 0.01f);
  bk.assign(d, 0.01f);
  bv.assign(d, 0.01f);
  gab.assign(static_cast<size_t>(t * t), 0.05f);
}

static void testGraphDense() {
  v13::InteractionGraph g = v13::buildDenseGraph(8);
  CHECK(g.numEdges() == 64);
  CHECK(g.isDense());
  auto aff = g.affectedEdges({3});
  CHECK(static_cast<int>(aff.size()) == 15);
  auto rows = g.affectedRows({5, 3, 3});
  CHECK(rows.size() == 8);
  CHECK(v13::scoreCellsForChanged(8, {3}) == 15);
  CHECK(v13::scoreCellsForChanged(8, {0, 5}) == 28);
}

static void testIncrParity() {
  const int T = 8, D = 32;
  std::vector<float> wq, bq, wk, bk, wv, bv, gab;
  fillWeights(T, D, wq, bq, wk, bk, wv, bv, gab);
  v13::IncrWeights w;
  w.wq = wq.data(); w.bq = bq.data(); w.wk = wk.data(); w.bk = bk.data();
  w.wv = wv.data(); w.bv = bv.data(); w.gab = gab.data();
  w.tokens = T; w.dim = D; w.alpha = 1.0f;
  v13::RelationalCache c;
  std::string err;
  CHECK(c.configure(w, 8, err));
  std::vector<float> x0(256), x1(256);
  for (int i = 0; i < 256; ++i) x0[i] = static_cast<float>((i * 37 % 100)) / 100.0f;
  x1 = x0;
  for (int d = 0; d < D; ++d) x1[3 * D + d] += 0.05f;
  c.rebuild(x0.data(), nullptr);
  c.update(x1.data(), nullptr, {3});
  float md = 0;
  CHECK(c.verifyAgainstFull(x1.data(), nullptr, 1e-5f, &md));
  std::vector<float> x2 = x1;
  for (int d = 0; d < D; ++d) { x2[d] += 0.1f; x2[5 * D + d] -= 0.1f; }
  c.update(x2.data(), nullptr, {0, 5});
  CHECK(c.verifyAgainstFull(x2.data(), nullptr, 1e-5f, &md));
}

static void testFallbackThreshold() {
  const int T = 8, D = 32;
  std::vector<float> wq, bq, wk, bk, wv, bv, gab;
  fillWeights(T, D, wq, bq, wk, bk, wv, bv, gab);
  v13::IncrWeights w;
  w.wq = wq.data(); w.bq = bq.data(); w.wk = wk.data(); w.bk = bk.data();
  w.wv = wv.data(); w.bv = bv.data(); w.gab = gab.data();
  w.tokens = T; w.dim = D;
  v13::RelationalCache c;
  std::string err;
  CHECK(c.configure(w, 1, err));
  std::vector<float> x0(256, 0.5f), x1(256, 0.5f);
  x1[100] = 0.7f; x1[200] = 0.2f;
  c.rebuild(x0.data(), nullptr);
  c.update(x1.data(), nullptr, {3});
  CHECK(c.incrementalUpdates() == 1);
  c.update(x1.data(), nullptr, {0, 5});
  CHECK(c.fallbacks() == 1);
  float md = 0;
  CHECK(c.verifyAgainstFull(x1.data(), nullptr, 1e-5f, &md));
  CHECK(!c.useIncremental({0, 1}));
  CHECK(c.useIncremental({0}));
}

static void testStackIsolation() {
  const int T = 8, D = 32;
  std::vector<float> wq, bq, wk, bk, wv, bv, gab;
  fillWeights(T, D, wq, bq, wk, bk, wv, bv, gab);
  v13::IncrWeights w;
  w.wq = wq.data(); w.bq = bq.data(); w.wk = wk.data(); w.bk = bk.data();
  w.wv = wv.data(); w.bv = bv.data(); w.gab = gab.data();
  w.tokens = T; w.dim = D;
  v13::RelationalCache c;
  std::string err;
  CHECK(c.configure(w, 8, err));
  std::vector<float> x0(256, 0.3f), xa(256, 0.3f), xb(256, 0.3f);
  xa[10] = 0.9f;
  xb[200] = 0.1f;
  c.rebuild(x0.data(), nullptr);
  c.push();
  c.update(xa.data(), nullptr, {0});
  float mda = 0;
  CHECK(c.verifyAgainstFull(xa.data(), nullptr, 1e-5f, &mda));
  c.pop();
  c.update(xb.data(), nullptr, {6});
  float mdb = 0;
  CHECK(c.verifyAgainstFull(xb.data(), nullptr, 1e-5f, &mdb));
  CHECK(std::fabs(c.tokens()[200] - 0.1f) < 1e-6);
}

static void testTokenDelta() {
  std::vector<ActiveFeature> before = {{0, 5}, {0, 9}, {1, 3}};
  std::vector<ActiveFeature> after = {{0, 5}, {0, 12}, {1, 3}};
  v13::GroupDelta d = v13::detectChangedGroups(before, after);
  CHECK(d.changedGroups.size() == 1);
  CHECK(d.changedGroups[0] == 0);
  int map[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  auto ct = v13::changedTokensFromGroups(d, map, 8);
  CHECK(ct.size() == 1);
  CHECK(ct[0] == 0);
  float a[4] = {1, 2, 3, 4};
  float b[4] = {1, 2, 9, 4};
  auto cmp = v13::changedTokensByCompare(a, b, 2, 2);
  CHECK(cmp.size() == 1);
  CHECK(cmp[0] == 1);
}

static void testAllTokenFallback() {
  const int T = 8, D = 32;
  std::vector<float> wq, bq, wk, bk, wv, bv, gab;
  fillWeights(T, D, wq, bq, wk, bk, wv, bv, gab);
  v13::IncrWeights w;
  w.wq = wq.data(); w.bq = bq.data(); w.wk = wk.data(); w.bk = bk.data();
  w.wv = wv.data(); w.bv = bv.data(); w.gab = gab.data();
  w.tokens = T; w.dim = D;
  v13::RelationalCache c;
  std::string err;
  CHECK(c.configure(w, 3, err));
  std::vector<float> x0(256, 0.2f), x1(256, 0.8f);
  c.rebuild(x0.data(), nullptr);
  c.update(x1.data(), nullptr, {0, 1, 2, 3, 4, 5, 6, 7});
  CHECK(c.fallbacks() == 1);
  float md = 0;
  CHECK(c.verifyAgainstFull(x1.data(), nullptr, 1e-5f, &md));
  auto bytes = c.cacheBytes();
  CHECK(bytes.scores == 256);
  CHECK(bytes.gates == 256);
  CHECK(bytes.mixed == 1024);
}

static bool findVectorFile(std::string& path) {
  const char* cands[] = {
      "spec/test-vectors/v13/incremental.txt",
      "../spec/test-vectors/v13/incremental.txt",
      "../../spec/test-vectors/v13/incremental.txt",
      "/workspaces/rune/spec/test-vectors/v13/incremental.txt",
  };
  for (const char* c : cands) {
    std::ifstream f(c);
    if (f.good()) {
      path = c;
      return true;
    }
  }
  return false;
}

static std::vector<float> parseLine(const std::string& line) {
  std::vector<float> out;
  std::istringstream ss(line);
  float v;
  while (ss >> v) out.push_back(v);
  return out;
}

static void testSharedVectors() {
  std::string path;
  if (!findVectorFile(path)) return;
  std::ifstream f(path);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty()) lines.push_back(line);
  }
  CHECK(lines.size() == 13);
  if (lines.size() != 13) return;
  int T, D, thr;
  char gate[32];
  float alpha;
  CHECK(std::sscanf(lines[0].c_str(), "%d %d %d %31s %f", &T, &D, &thr, gate, &alpha) == 5);
  std::vector<float> wq = parseLine(lines[1]);
  std::vector<float> bq = parseLine(lines[2]);
  std::vector<float> wk = parseLine(lines[3]);
  std::vector<float> bk = parseLine(lines[4]);
  std::vector<float> wv = parseLine(lines[5]);
  std::vector<float> bv = parseLine(lines[6]);
  std::vector<float> gab = parseLine(lines[7]);
  std::vector<float> x0 = parseLine(lines[8]);
  std::vector<float> x1 = parseLine(lines[9]);
  std::vector<float> expFull = parseLine(lines[10]);
  std::vector<float> expIncr = parseLine(lines[11]);
  std::vector<int> changed;
  {
    std::istringstream ss(lines[12]);
    int c;
    while (ss >> c) changed.push_back(c);
  }
  CHECK(static_cast<int>(wq.size()) == D * D);
  CHECK(static_cast<int>(x0.size()) == T * D);
  v13::IncrWeights w;
  w.wq = wq.data(); w.bq = bq.data(); w.wk = wk.data(); w.bk = bk.data();
  w.wv = wv.data(); w.bv = bv.data(); w.gab = gab.data();
  w.tokens = T; w.dim = D; w.alpha = alpha;
  v13::RelationalCache c;
  std::string err;
  CHECK(c.configure(w, thr, err));
  c.rebuild(x0.data(), nullptr);
  c.update(x1.data(), nullptr, changed);
  const auto& got = c.out();
  CHECK(static_cast<int>(got.size()) == static_cast<int>(expFull.size()));
  for (size_t i = 0; i < got.size(); ++i) {
    CHECK_CLOSE(static_cast<double>(got[i]), static_cast<double>(expFull[i]), 1e-5);
    CHECK_CLOSE(static_cast<double>(got[i]), static_cast<double>(expIncr[i]), 1e-5);
  }
  float md = 0;
  CHECK(c.verifyAgainstFull(x1.data(), nullptr, 1e-5f, &md));
}

static void testDynamicBiasParity() {
  const int T = 8, D = 32, C = 8;
  std::vector<float> wq, bq, wk, bk, wv, bv, gab;
  fillWeights(T, D, wq, bq, wk, bk, wv, bv, gab);
  std::vector<float> dynU(T * C, 0.02f), dynW(T * C, 0.02f);
  v13::IncrWeights w;
  w.wq = wq.data(); w.bq = bq.data(); w.wk = wk.data(); w.bk = bk.data();
  w.wv = wv.data(); w.bv = bv.data(); w.gab = gab.data();
  w.dynU = dynU.data(); w.dynW = dynW.data();
  w.tokens = T; w.dim = D; w.ctxDim = C;
  v13::RelationalCache c;
  std::string err;
  CHECK(c.configure(w, 8, err));
  std::vector<float> x0(256, 0.4f), x1(256, 0.4f), ctx(C, 0.5f);
  x1[35] = 0.9f;
  c.rebuild(x0.data(), ctx.data());
  c.update(x1.data(), ctx.data(), {1});
  float md = 0;
  CHECK(c.verifyAgainstFull(x1.data(), ctx.data(), 1e-5f, &md));
}

static void testDynCtxChange() {
  const int T = 8, D = 32, C = 8;
  std::vector<float> wq, bq, wk, bk, wv, bv, gab;
  fillWeights(T, D, wq, bq, wk, bk, wv, bv, gab);
  std::vector<float> dynU(T * C, 0.02f), dynW(T * C, 0.03f);
  v13::IncrWeights w;
  w.wq = wq.data(); w.bq = bq.data(); w.wk = wk.data(); w.bk = bk.data();
  w.wv = wv.data(); w.bv = bv.data(); w.gab = gab.data();
  w.dynU = dynU.data(); w.dynW = dynW.data();
  w.tokens = T; w.dim = D; w.ctxDim = C;
  v13::RelationalCache c;
  std::string err;
  CHECK(c.configure(w, 8, err));
  std::vector<float> x0(256, 0.4f), x1(256, 0.4f), ctx0(C, 0.5f), ctx1(C, 0.9f);
  x1[35] = 0.9f;
  c.rebuild(x0.data(), ctx0.data());
  c.update(x0.data(), ctx1.data(), {});
  float md = 0;
  CHECK(c.verifyAgainstFull(x0.data(), ctx1.data(), 1e-5f, &md));
  c.update(x0.data(), nullptr, {});
  CHECK(c.verifyAgainstFull(x0.data(), nullptr, 1e-5f, &md));
  c.update(x1.data(), nullptr, {1});
  CHECK(c.verifyAgainstFull(x1.data(), nullptr, 1e-5f, &md));
}

static void testPopEmptyAndDynRestore() {
  const int T = 8, D = 32, C = 8;
  std::vector<float> wq, bq, wk, bk, wv, bv, gab;
  fillWeights(T, D, wq, bq, wk, bk, wv, bv, gab);
  std::vector<float> dynU(T * C, 0.02f), dynW(T * C, 0.03f);
  v13::IncrWeights w;
  w.wq = wq.data(); w.bq = bq.data(); w.wk = wk.data(); w.bk = bk.data();
  w.wv = wv.data(); w.bv = bv.data(); w.gab = gab.data();
  w.dynU = dynU.data(); w.dynW = dynW.data();
  w.tokens = T; w.dim = D; w.ctxDim = C;
  v13::RelationalCache c;
  std::string err;
  CHECK(c.configure(w, 8, err));
  CHECK(!c.pop());
  std::vector<float> x0(256, 0.4f), x1(256, 0.4f), ctx0(C, 0.5f), ctx1(C, 0.9f);
  x1[35] = 0.9f;
  c.rebuild(x0.data(), ctx0.data());
  c.push();
  c.update(x1.data(), ctx1.data(), {1});
  CHECK(c.pop());
  float md = 0;
  CHECK(c.verifyAgainstFull(x0.data(), ctx0.data(), 1e-5f, &md));
  CHECK(!c.pop());
}

void runV13Tests() {
  testGraphDense();
  testSharedVectors();
  testDynamicBiasParity();
  testDynCtxChange();
  testPopEmptyAndDynRestore();
  testIncrParity();
  testFallbackThreshold();
  testStackIsolation();
  testTokenDelta();
  testAllTokenFallback();
}
