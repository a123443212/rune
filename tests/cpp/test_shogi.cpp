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

#include <cctype>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "core/architectures/mlp/mlp.h"
#include "core/model_io/model_io.h"
#include "core/shogi/shogi_board.h"
#include "core/shogi/shogi_evaluator.h"
#include "core/shogi/shogi_features.h"
#include "tests/cpp/test_framework.h"

using namespace rune;
using namespace rune::shogi;

namespace {

bool hasFeature(const std::vector<ActiveFeature>& feats, int g, int idx) {
  for (const auto& f : feats) {
    if (f.group == g && f.index == idx) return true;
  }
  return false;
}

void testShogiParse() {
  ShogiBoard b;
  CHECK(b.setSfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"));
  CHECK(b.sideToMove() == kBlack);
  CHECK(b.moveNo() == 1);
  ShogiPiece k = b.at(0);
  CHECK(k.present && k.kind == kLance && k.color == kWhite);
  ShogiBoard w;
  CHECK(w.setSfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w 2P3p 25"));
  CHECK(w.sideToMove() == kWhite);
  CHECK(w.hand(kBlack, 0) == 2);
  CHECK(w.hand(kWhite, 0) == 3);
  CHECK(w.moveNo() == 25);
  ShogiBoard bad;
  CHECK(!bad.setSfen("8/8 w - 1"));
  CHECK(!bad.setSfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSN b - 1"));
  CHECK(!bad.setSfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL x - 1"));
}

void testShogiExtractStartpos() {
  ShogiBoard b("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
  std::vector<ActiveFeature> feats;
  ShogiFeatureSet::extract(b, feats);
  CHECK(feats.size() == 96);
  for (size_t i = 1; i < feats.size(); ++i) CHECK(feats[i - 1] < feats[i]);
  CHECK(hasFeature(feats, 7, 0));
  CHECK(hasFeature(feats, 7, 2));
  CHECK(hasFeature(feats, 7, 10));
  CHECK(hasFeature(feats, 7, 14));
  CHECK(hasFeature(feats, 3, 0));
  CHECK(hasFeature(feats, 3, 19));
  CHECK(hasFeature(feats, 0, 54));
  CHECK(hasFeature(feats, 1, 75));
  float ctx[12];
  ShogiFeatureSet::context(b, ctx);
  CHECK(ctx[0] == 0.0f);
  CHECK(ctx[8] == 0.0f);
  CHECK(ShogiFeatureSet::phase(b) == 0);
}

void testShogiCheckAndHands() {
  ShogiBoard b("lnsgkgsn1/1r5b1/pppp1pppp/4p4/9/4P4/PPPP1PPPP/1B5R1/LNSGKGSNL b 2P 10");
  std::vector<ActiveFeature> feats;
  ShogiFeatureSet::extract(b, feats);
  CHECK(hasFeature(feats, 3, 2));
  ShogiBoard c("4k4/9/9/9/9/9/9/9/4R4 w - 1");
  std::vector<ActiveFeature> cf;
  ShogiFeatureSet::extract(c, cf);
  bool foundChecker = false;
  for (const auto& f : cf) {
    if (f.group == 5) foundChecker = true;
  }
  CHECK(foundChecker);
  CHECK(hasFeature(cf, 7, 13));
}

void testShogiEvalGolden() {
  std::ifstream f("spec/test-vectors/shogi/eval.json");
  CHECK(!!f);
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  RuneFile rf;
  std::string err;
  CHECK(loadRuneFile("spec/test-vectors/models/shogi-mlp-fp32.rune", rf, err));
  CHECK(rf.spec.game == "shogi");
  CHECK(rf.spec.featureSet == "shogi_raw_v01");
  int count = 0;
  size_t p = 0;
  while (true) {
    size_t sp = s.find("\"sfen\"", p);
    if (sp == std::string::npos) break;
    size_t q0 = s.find('"', sp + 6) + 1;
    size_t q1 = s.find('"', q0);
    std::string sfen = s.substr(q0, q1 - q0);
    size_t vp = s.find("\"value\"", q1);
    size_t vv = vp + 8;
    while (vv < s.size() && (s[vv] == ' ' || s[vv] == '\n' || s[vv] == '\r' || s[vv] == '\t' ||
                             s[vv] == ':' || s[vv] == ',')) ++vv;
    size_t ve = vv;
    while (ve < s.size() && (std::isdigit(s[ve]) || s[ve] == '.' || s[ve] == '-' ||
                             s[ve] == '+' || s[ve] == 'e' || s[ve] == 'E')) ++ve;
    double value = std::stod(s.substr(vv, ve - vv));
    size_t wp = s.find("\"wdl\"", ve);
    size_t wv = wp + 6;
    double wdl[3];
    for (int i = 0; i < 3; ++i) {
      while (wv < s.size() && (s[wv] == ' ' || s[wv] == '\n' || s[wv] == '\r' || s[wv] == '\t' ||
                               s[wv] == ':' || s[wv] == '[' || s[wv] == ',')) ++wv;
      size_t we = wv;
      while (we < s.size() && (std::isdigit(s[we]) || s[we] == '.' || s[we] == '-' ||
                               s[we] == '+' || s[we] == 'e' || s[we] == 'E')) ++we;
      wdl[i] = std::stod(s.substr(wv, we - wv));
      wv = we;
    }
    p = wv;
    ShogiEvaluator ev(&rf.embeddings, rf.arch.get());
    EvalResult r;
    CHECK(ev.evaluateSfen(sfen, r));
    CHECK(std::fabs(r.value - value) < 1e-5);
    for (int i = 0; i < 3; ++i) CHECK(std::fabs(r.wdl[i] - wdl[i]) < 1e-5);
    ++count;
  }
  CHECK(count == 5);
}

void testShogiIncrementalMatchesRefresh() {
  const char* sb = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
  const char* sc = "lnsgkgsn1/1r5b1/pppp1pppp/4p4/9/4P4/PPPP1PPPP/1B5R1/LNSGKGSNL b 2P 10";
  ShogiBoard b(sb);
  ShogiBoard c(sc);
  std::vector<ActiveFeature> before, after, added, removed;
  ShogiFeatureSet::extract(b, before);
  ShogiFeatureSet::extract(c, after);
  GroupedFeatureSet::diffFeatures(before, after, added, removed);
  int vocabs[9];
  for (int g = 0; g < 9; ++g) vocabs[g] = ShogiFeatureSet::vocabSize(g);
  EmbeddingTables tables(vocabs);
  tables.init(3);
  GroupedMlp mlp;
  ShogiEvaluator ev(&tables, &mlp);
  CHECK(ev.refresh(sb));
  ev.updateIncremental(added, removed);
  EvalResult rInc = ev.evaluate();
  ShogiEvaluator ev2(&tables, &mlp);
  CHECK(ev2.refresh(sc));
  EvalResult rFull = ev2.evaluate();
  CHECK_CLOSE(rInc.value, rFull.value, 1e-5);
}

}  // namespace

void testShogi() {
  testShogiParse();
  testShogiExtractStartpos();
  testShogiCheckAndHands();
  testShogiEvalGolden();
  testShogiIncrementalMatchesRefresh();
}
