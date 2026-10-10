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
static float testMaterial(Board& b) {
  int us = static_cast<int>(b.sideToMove());
  int score = 0;
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = b.at(sq);
    if (p.empty()) continue;
    int v = 0;
    if (p.type == PieceType::Pawn) v = 100;
    else if (p.type == PieceType::Knight) v = 320;
    else if (p.type == PieceType::Bishop) v = 330;
    else if (p.type == PieceType::Rook) v = 500;
    else if (p.type == PieceType::Queen) v = 900;
    score += (static_cast<int>(p.color) == us) ? v : -v;
  }
  return static_cast<float>(score) / 1000.0f;
}
static void testSearchQuiescence() {
  Board b("r3k3/8/8/8/8/8/8/R3K3 w - - 0 1");
  eng::LazyConfig cfg;
  eng::EvalFn ev = [](Board& x) -> float { return testMaterial(x); };
  eng::SearchStats st;
  float s = eng::searchRoot(b, 0, ev, cfg, st, -1e9f, 1e9f);
  CHECK(st.hasMove);
  CHECK(st.qnodes > 0);
  CHECK(s > 0.4f);
}
static void testSearchLazy() {
  Board b;
  eng::LazyConfig cfg;
  cfg.mode = eng::LazyMode::L1;
  cfg.threshold = 0.5f;
  int fullCalls = 0;
  eng::EvalFn cheap = [](Board&) -> float { return 0.0f; };
  eng::EvalFn full = [&fullCalls](Board& x) -> float {
    ++fullCalls;
    std::vector<Move> lm;
    x.generateLegalMoves(lm);
    return 0.05f;
  };
  eng::SearchStats st;
  float s = eng::searchRootLazy(b, 2, cheap, full, cfg, st, -1e9f, 1e9f);
  CHECK(st.hasMove);
  CHECK(fullCalls == 0);
  CHECK(st.refined == 0);
  CHECK(s > -2.0f && s < 2.0f);
  eng::SearchStats st2;
  eng::EvalFn cheapHigh = [](Board&) -> float { return 0.9f; };
  float s2 = eng::searchRootLazy(b, 1, cheapHigh, full, cfg, st2, -1e9f, 1e9f);
  CHECK(st2.refined > 0);
  CHECK(fullCalls > 0);
  CHECK(s2 > -2.0f && s2 < 2.0f);
}
static void testSearchTt() {
  Board b;
  eng::LazyConfig cfg;
  eng::EvalFn ev = [](Board& x) -> float {
    std::vector<Move> lm;
    x.generateLegalMoves(lm);
    return 0.05f;
  };
  eng::SearchStats st;
  eng::searchRoot(b, 3, ev, cfg, st, -1e9f, 1e9f);
  CHECK(st.ttHits > 0);
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
  Board fifty("6k1/8/8/8/8/8/8/K6R w - - 100 45");
  eng::SearchStats st3;
  float s3 = eng::searchRoot(fifty, 2, ev, cfg, st3, -1e9f, 1e9f);
  CHECK(st3.hasMove);
  CHECK(std::fabs(s3) < 1e-6f);
}
static void testMoveOrdering() {
  Board b("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
  std::vector<Move> moves;
  b.generateLegalMoves(moves);
  CHECK(!moves.empty());
  eng::orderMoves(b, moves);
  CHECK(!b.at(moves.front().to).empty());
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
  testMoveOrdering();
  testSearchQuiescence();
  testSearchLazy();
  testSearchTt();
  testIncrementalStress();
  testThreadStates();
}
