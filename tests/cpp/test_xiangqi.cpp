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
#include "core/xiangqi/xiangqi_board.h"
#include "core/xiangqi/xiangqi_evaluator.h"
#include "core/xiangqi/xiangqi_features.h"
#include "tests/cpp/test_framework.h"

using namespace rune;
using namespace rune::xiangqi;

namespace {

bool hasFeature(const std::vector<ActiveFeature>& feats, int g, int idx) {
  for (const auto& f : feats) {
    if (f.group == g && f.index == idx) return true;
  }
  return false;
}

void testXiangqiParse() {
  XiangqiBoard b;
  CHECK(b.setFen("rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1"));
  CHECK(b.sideToMove() == kRed);
  CHECK(b.moveNo() == 1);
  XiangqiPiece k = b.at(0);
  CHECK(k.present && k.kind == kRook && k.color == kRed);
  XiangqiPiece kb = b.at(81);
  CHECK(kb.present && kb.kind == kRook && kb.color == kBlack);
  XiangqiBoard w;
  CHECK(w.setFen("rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR b - - 3 25"));
  CHECK(w.sideToMove() == kBlack);
  CHECK(w.moveNo() == 25);
  XiangqiBoard bad;
  CHECK(!bad.setFen("rheakaehr/9 w - - 0 1"));
  CHECK(!bad.setFen("rheakaeh/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1"));
  CHECK(!bad.setFen("rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR x - - 0 1"));
}

void testXiangqiExtractStartpos() {
  XiangqiBoard b("rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1");
  std::vector<ActiveFeature> feats;
  XiangqiFeatureSet::extract(b, feats);
  CHECK(feats.size() == 78);
  for (size_t i = 1; i < feats.size(); ++i) CHECK(feats[i - 1] < feats[i]);
  CHECK(hasFeature(feats, 7, 0));
  CHECK(hasFeature(feats, 7, 2));
  CHECK(hasFeature(feats, 7, 5));
  CHECK(hasFeature(feats, 7, 23));
  CHECK(hasFeature(feats, 0, 27));
  CHECK(hasFeature(feats, 3, 1));
  float ctx[12];
  XiangqiFeatureSet::context(b, ctx);
  CHECK(ctx[0] == 0.0f);
  CHECK(ctx[7] == 0.0f);
  CHECK(XiangqiFeatureSet::phase(b) == 0);
}

void testXiangqiCheckAndFlying() {
  XiangqiBoard b("4k4/4a4/9/9/4r4/9/9/9/9/4K4 w - - 0 1");
  std::vector<ActiveFeature> feats;
  XiangqiFeatureSet::extract(b, feats);
  CHECK(hasFeature(feats, 7, 21));
  XiangqiBoard c("4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1");
  std::vector<ActiveFeature> cf;
  XiangqiFeatureSet::extract(c, cf);
  CHECK(hasFeature(cf, 7, 21));
  CHECK(hasFeature(cf, 7, 22));
}

void testXiangqiEvalGolden() {
  std::ifstream f("spec/test-vectors/xiangqi/eval.json");
  CHECK(!!f);
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  RuneFile rf;
  std::string err;
  CHECK(loadRuneFile("spec/test-vectors/models/xiangqi-mlp-fp32.rune", rf, err));
  CHECK(rf.spec.game == "xiangqi");
  CHECK(rf.spec.featureSet == "xiangqi_raw_v01");
  int count = 0;
  size_t p = 0;
  while (true) {
    size_t sp = s.find("\"fen\"", p);
    if (sp == std::string::npos) break;
    size_t q0 = s.find('"', sp + 5) + 1;
    size_t q1 = s.find('"', q0);
    std::string fen = s.substr(q0, q1 - q0);
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
    XiangqiEvaluator ev(&rf.embeddings, rf.arch.get());
    EvalResult r;
    CHECK(ev.evaluateFen(fen, r));
    CHECK(std::fabs(r.value - value) < 1e-5);
    for (int i = 0; i < 3; ++i) CHECK(std::fabs(r.wdl[i] - wdl[i]) < 1e-5);
    ++count;
  }
  CHECK(count == 5);
}

void testXiangqiIncrementalMatchesRefresh() {
  const char* sb = "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1";
  const char* sc = "2eak4/4k4/2h1a4/p3p1p1p/4c4/4P4/P1P1C1P1P/9/9/RHEAKAEHR w - - 0 1";
  XiangqiBoard b(sb);
  XiangqiBoard c(sc);
  std::vector<ActiveFeature> before, after, added, removed;
  XiangqiFeatureSet::extract(b, before);
  XiangqiFeatureSet::extract(c, after);
  GroupedFeatureSet::diffFeatures(before, after, added, removed);
  int vocabs[9];
  for (int g = 0; g < 9; ++g) vocabs[g] = XiangqiFeatureSet::vocabSize(g);
  EmbeddingTables tables(vocabs);
  tables.init(3);
  GroupedMlp mlp;
  XiangqiEvaluator ev(&tables, &mlp);
  CHECK(ev.refresh(sb));
  ev.updateIncremental(added, removed);
  EvalResult rInc = ev.evaluate();
  XiangqiEvaluator ev2(&tables, &mlp);
  CHECK(ev2.refresh(sc));
  EvalResult rFull = ev2.evaluate();
  CHECK_CLOSE(rInc.value, rFull.value, 1e-5);
}

}  // namespace

void testXiangqi() {
  testXiangqiParse();
  testXiangqiExtractStartpos();
  testXiangqiCheckAndFlying();
  testXiangqiEvalGolden();
  testXiangqiIncrementalMatchesRefresh();
}
