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

#include <cmath>
#include <string>
#include <vector>

#include "core/architectures/mlp/mlp.h"
#include "core/go/go_board.h"
#include "core/go/go_evaluator.h"
#include "core/go/go_features.h"
#include "core/go/go_planes.h"
#include "core/go/go_resnet.h"
#include "core/kernels/conv_kernels.h"
#include "core/kernels/policy_kernels.h"
#include "tests/cpp/test_framework.h"

using namespace rune;
using namespace rune::go;

namespace {

bool hasFeature(const std::vector<ActiveFeature>& feats, int g, int idx) {
  for (const auto& f : feats) {
    if (f.group == g && f.index == idx) return true;
  }
  return false;
}

std::string emptyState(int n) {
  std::string grid;
  for (int r = 0; r < n; ++r) {
    if (r) grid += "/";
    grid += std::string(static_cast<size_t>(n), '.');
  }
  return grid + " b";
}

void testGoParse() {
  GoBoard b;
  CHECK(b.setState(emptyState(9)));
  CHECK(b.size() == 9);
  CHECK(b.sideToMove() == kBlack);
  GoBoard w;
  CHECK(w.setState(emptyState(19)));
  CHECK(w.size() == 19);
  GoBoard bad;
  CHECK(!bad.setState("..../.... b"));
  CHECK(!bad.setState("........./......... b"));
  GoBoard stone;
  CHECK(stone.setState("........./........./........./....X..../........./........./........./........./......... b"));
  CHECK(stone.at(3, 4) == 1);
  CHECK(stone.atSq(3 * 9 + 4) == 1);
  CHECK(stone.libertiesOf(3, 4) == 4);
  CHECK(stone.libertiesOf(0, 0) == 0);
}

void testGoExtract() {
  GoBoard b;
  CHECK(b.setState("........./........./........./....X..../........./........./........./........./......... b"));
  std::vector<ActiveFeature> feats;
  GoFeatureSet::extract(b, feats);
  CHECK(!feats.empty());
  for (size_t i = 1; i < feats.size(); ++i) CHECK(feats[i - 1] < feats[i]);
  CHECK(hasFeature(feats, 7, 0));
  CHECK(hasFeature(feats, 7, 2));
  float ctx[12];
  GoFeatureSet::context(b, ctx);
  CHECK(ctx[0] == 0.0f);
  CHECK(GoFeatureSet::phase(b) == 0);
  CHECK(GoFeatureSet::vocabSize(0) == 722);
}

void testGoPlanes() {
  GoBoard b;
  CHECK(b.setState("........./........./........./....X..../........./........./........./........./......... b"));
  std::vector<float> planes;
  extractPlanes(b, planes);
  CHECK(planes.size() == 81);
  CHECK(planes[3 * 9 + 4] == 1.0f);
  CHECK(planes[0] == 0.0f);
  GoBoard w;
  CHECK(w.setState("........./........./........./....X..../........./........./........./........./......... w"));
  std::vector<float> pw;
  extractPlanes(w, pw);
  CHECK(pw[3 * 9 + 4] == -1.0f);
}

void testGoKernels() {
  float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float w[1] = {1.0f};
  float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  conv::conv2dNchw(x, w, nullptr, out, 1, 1, 1, 2, 2, 1, 1, 0, 0);
  CHECK_CLOSE(out[0], 1.0, 1e-6);
  CHECK_CLOSE(out[3], 4.0, 1e-6);
  float logits[3] = {1.0f, 2.0f, 3.0f};
  float probs[3] = {0.0f, 0.0f, 0.0f};
  policy::softmax(logits, probs, 3);
  CHECK_CLOSE(probs[0] + probs[1] + probs[2], 1.0, 1e-6);
  CHECK(probs[2] > probs[1] && probs[1] > probs[0]);
}

void testGoResnetDeterministic() {
  GoResnetSizes sz;
  sz.board = 9;
  sz.channels = 2;
  sz.blocks = 1;
  sz.policySize = 82;
  sz.valueH2 = 4;
  GoResnetWeights wt;
  wt.stemW.assign(2 * 1 * 3 * 3, 0.05f);
  wt.stemB.assign(2, 0.0f);
  wt.blockW1.assign(1, std::vector<float>(2 * 2 * 3 * 3, 0.02f));
  wt.blockB1.assign(1, std::vector<float>(2, 0.0f));
  wt.blockW2.assign(1, std::vector<float>(2 * 2 * 3 * 3, 0.02f));
  wt.blockB2.assign(1, std::vector<float>(2, 0.0f));
  wt.vh1.assign(4 * 2, 0.1f);
  wt.bh1.assign(4, 0.0f);
  wt.wv.assign(4, 0.25f);
  wt.bv = 0.0f;
  wt.wwdl.assign(3 * 4, 0.1f);
  wt.bwdl.assign(3, 0.0f);
  wt.wpol.assign(82 * 2 * 81, 0.001f);
  wt.bpol.assign(82, 0.0f);
  GoBoard b;
  CHECK(b.setState(emptyState(9)));
  std::vector<float> planes;
  extractPlanes(b, planes);
  GoResnetOutput r1 = forwardGoResnet(wt, sz, planes.data());
  GoResnetOutput r2 = forwardGoResnet(wt, sz, planes.data());
  CHECK_CLOSE(r1.value, r2.value, 1e-9);
  GoResnetScratch sc;
  GoResnetOutput r3 = forwardGoResnetFast(wt, sz, planes.data(), sc);
  CHECK_CLOSE(r1.value, r3.value, 1e-5);
  for (size_t i = 0; i < r1.policy.size(); ++i) CHECK_CLOSE(r1.policy[i], r3.policy[i], 1e-5);
  CHECK(r1.policy.size() == 82);
  float s = 0.0f;
  for (float v : r1.policy) s += v;
  CHECK_CLOSE(s, 1.0, 1e-5);
}

void testGoEvalClassic() {
  int vocabs[9];
  for (int g = 0; g < 9; ++g) vocabs[g] = GoFeatureSet::vocabSize(g);
  EmbeddingTables tables(vocabs);
  tables.init(3);
  GroupedMlp mlp;
  GoEvaluator ev(&tables, &mlp);
  EvalResult r;
  CHECK(ev.evaluateState(emptyState(9), r));
  CHECK(r.value >= -1.0f && r.value <= 1.0f);
}

}

void testGo() {
  testGoParse();
  testGoExtract();
  testGoPlanes();
  testGoKernels();
  testGoResnetDeterministic();
  testGoEvalClassic();
}
