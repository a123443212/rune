#include "tests/cpp/test_v12.h"
#include "tests/cpp/test_framework.h"
#include <cmath>
#include <vector>
#include "core/board/board.h"
#include "core/engine/eval_cache.h"
#include "core/engine/eval_contract.h"
#include "core/engine/search.h"
#include "core/accumulators/grouped_accumulator.h"
#include "core/features/feature_set.h"
#include <thread>
using namespace rune;
static void testScale() {
  CHECK(eng::canonicalValue(2.0f) == 1.0f);
  CHECK(eng::canonicalValue(-2.0f) == -1.0f);
  CHECK(eng::canonicalValue(0.0f) == 0.0f);
  CHECK(eng::engineScore(1.0f) == 1000);
  CHECK(eng::engineScore(-1.0f) == -1000);
  CHECK(eng::engineScore(2.0f) == 1000);
  float w[3] = {0.5f, 0.3f, 0.2f};
  CHECK(std::fabs(eng::wdlValue(w) - 0.3f) < 1e-6);
  CHECK(eng::isExtreme(0.99f, 0.95f));
  CHECK(!eng::isExtreme(0.1f, 0.95f));
}
static void testStm() {
  Board a("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  Board b("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
  CHECK(a.sideToMove() != b.sideToMove());
  CHECK(a.toFen() != b.toFen());
  std::string ka = eng::evalCacheKey(a.toFen(), "h", "full");
  std::string kb = eng::evalCacheKey(b.toFen(), "h", "full");
  CHECK(ka != kb);
}
static void testCache() {
  eng::EvalCache c("modelA", "full", 4);
  float v = 0.0f;
  CHECK(!c.get("fen1", v));
  c.put("fen1", 0.5f);
  CHECK(c.get("fen1", v));
  CHECK(std::fabs(v - 0.5f) < 1e-9);
  eng::EvalCache d("modelB", "full", 4);
  CHECK(!d.get("fen1", v));
  std::string ka = eng::evalCacheKey("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "h", "full");
  std::string kb = eng::evalCacheKey("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "h", "full");
  CHECK(ka != kb);
  eng::EvalCache z("modelZ", "full", 0);
  for (int i = 0; i < 100; ++i) z.put("fen" + std::to_string(i), 0.5f);
  CHECK(z.size() == 0);
  CHECK(!z.get("fen0", v));
}
static void testSearchRuns() {
  Board b;
  eng::LazyConfig cfg;
  eng::EvalFn ev = [](Board& x) -> float {
    std::vector<Move> lm;
    x.generateLegalMoves(lm);
    return 0.05f;
  };
  eng::SearchStats st;
  float s = eng::searchRoot(b, 2, ev, cfg, st, -1e9f, 1e9f);
  CHECK(st.nodes > 0);
  CHECK(st.hasMove);
  CHECK(s > -2.0f && s < 2.0f);
}
static void testSearchMate() {
  eng::LazyConfig cfg;
  eng::EvalFn ev = [](Board&) -> float { return 0.5f; };
  eng::SearchStats st;
  Board mate("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1");
  float s = eng::searchRoot(mate, 2, ev, cfg, st, -1e9f, 1e9f);
  CHECK(!st.hasMove);
  CHECK(s < -9000.0f);
  Board stale("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
  eng::SearchStats st2;
  float s2 = eng::searchRoot(stale, 2, ev, cfg, st2, -1e9f, 1e9f);
  CHECK(!st2.hasMove);
  CHECK(std::fabs(s2) < 1e-6f);
}
static void testIncrementalStress() {
  Board b;
  std::vector<ActiveFeature> f0;
  GroupedFeatureSet::extract(b, f0);
  EmbeddingTables t;
  t.init(7);
  GroupedAccumulator acc(&t);
  acc.refresh(f0);
  float ref[256];
  acc.tokens(ref);
  for (int i = 0; i < 200; ++i) {
    std::vector<Move> lm;
    b.generateLegalMoves(lm);
    if (lm.empty()) break;
    Board c = b;
    c.makeMove(lm[0]);
    std::vector<ActiveFeature> f1;
    GroupedFeatureSet::extract(c, f1);
    std::vector<ActiveFeature> added, removed;
    GroupedFeatureSet::diffFeatures(f0, f1, added, removed);
    GroupedAccumulator inc(&t);
    inc.refresh(f0);
    inc.applyDiff(added, removed);
    float got[256];
    inc.tokens(got);
    GroupedAccumulator fresh(&t);
    fresh.refresh(f1);
    float want[256];
    fresh.tokens(want);
    for (int k = 0; k < 256; ++k) CHECK(std::fabs(got[k] - want[k]) < 1e-6f);
    b = c;
    f0 = f1;
  }
}
static void testThreadStates() {
  auto worker = [](int seed) -> float {
    Board b;
    eng::LazyConfig cfg;
    eng::EvalFn ev = [seed](Board& x) -> float {
      std::vector<Move> lm;
      x.generateLegalMoves(lm);
      return (float)(((int)lm.size() + seed) % 7) * 0.05f;
    };
    eng::SearchStats st;
    return eng::searchRoot(b, 2, ev, cfg, st, -1e9f, 1e9f);
  };
  float r0 = worker(1);
  std::vector<float> out(4, 0.0f);
  std::vector<std::thread> th;
  for (int i = 0; i < 4; ++i) th.emplace_back([&, i]() { out[i] = worker(1); });
  for (auto& t : th) t.join();
  for (int i = 0; i < 4; ++i) CHECK(std::fabs(out[i] - r0) < 1e-9f);
}
void runV12Tests() {
  testScale();
  testStm();
  testCache();
  testSearchRuns();
  testSearchMate();
  testIncrementalStress();
  testThreadStates();
}
