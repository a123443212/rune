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
#include <vector>

#include "core/accumulators/grouped_accumulator.h"
#include "core/architectures/attention/attention.h"
#include "core/architectures/mlp/mlp.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"
#include "core/simd/simd.h"
#include "tests/cpp/test_framework.h"

using namespace rune;

static float clipped(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

void testAttentionMath() {
  RuneAttentionBlock blk(true);
  for (float& g : blk.gab) g = 0.25f;
  float x[256];
  for (int i = 0; i < 256; ++i) x[i] = static_cast<float>((i % 17) - 8) * 0.05f;
  float out[256];
  blk.forward(x, out);
  float q[256], k[256], v[256];
  for (int t = 0; t < 8; ++t) {
    simd::matVec(blk.wq.data(), x + t * 32, blk.bq.data(), q + t * 32, 32, 32);
    simd::matVec(blk.wk.data(), x + t * 32, blk.bk.data(), k + t * 32, 32, 32);
    simd::matVec(blk.wv.data(), x + t * 32, blk.bv.data(), v + t * 32, 32, 32);
  }
  for (int a = 0; a < 8; ++a) {
    for (int d = 0; d < 32; ++d) {
      float expect = x[a * 32 + d];
      for (int c = 0; c < 8; ++c) {
        float s = blk.gab[a * 8 + c];
        for (int dd = 0; dd < 32; ++dd) s += q[a * 32 + dd] * k[c * 32 + dd];
        expect += clipped(s) * v[c * 32 + d];
      }
      CHECK_CLOSE(out[a * 32 + d], expect, 1e-4);
    }
  }
  RuneAttentionBlock noGab(false);
  noGab.wq = blk.wq;
  noGab.bq = blk.bq;
  noGab.wk = blk.wk;
  noGab.bk = blk.bk;
  noGab.wv = blk.wv;
  noGab.bv = blk.bv;
  float outNoGab[256];
  noGab.forward(x, outNoGab);
  bool differs = false;
  for (int i = 0; i < 256; ++i) {
    if (std::fabs(out[i] - outNoGab[i]) > 1e-6) differs = true;
  }
  CHECK(differs);
}

void testSerialization() {
  GroupedMlp mlp;
  std::vector<std::string> names;
  std::vector<std::vector<int>> shapes;
  std::vector<const float*> data;
  mlp.getTensors(names, shapes, data);
  std::vector<float> flat;
  for (size_t i = 0; i < data.size(); ++i) {
    size_t n = 1;
    for (int s : shapes[i]) n *= static_cast<size_t>(s);
    for (size_t j = 0; j < n; ++j) flat.push_back(data[i][j]);
  }
  GroupedMlp mlp2;
  CHECK(mlp2.setTensors(names, flat));
  float tok[256];
  for (int i = 0; i < 256; ++i) tok[i] = static_cast<float>(i) / 512.0f;
  float v1, v2, w1[3], w2[3];
  mlp.forward(tok, v1, w1);
  mlp2.forward(tok, v2, w2);
  CHECK_CLOSE(v1, v2, 1e-7);
  for (int i = 0; i < 3; ++i) CHECK_CLOSE(w1[i], w2[i], 1e-7);
  CHECK(mlp.parameterCount() == flat.size());
  RuneAttnModel attn(true);
  CHECK(attn.parameterCount() > mlp.parameterCount());
  SfnnBaseline sfnn;
  CHECK(sfnn.parameterCount() > mlp.parameterCount());
  CHECK(attn.spec().configHash() != mlp.spec().configHash());
  CHECK(attn.spec().configHash() == attn.spec().configHash());
}

void testPairHead() {
  GroupedMlp mlp;
  std::vector<std::string> names;
  std::vector<std::vector<int>> shapes;
  std::vector<const float*> data;
  mlp.getTensors(names, shapes, data);
  std::vector<float> flatSingle;
  for (size_t i = 0; i < data.size(); ++i) {
    size_t n = 1;
    for (int s : shapes[i]) n *= static_cast<size_t>(s);
    for (size_t j = 0; j < n; ++j) flatSingle.push_back(data[i][j]);
  }
  size_t w1n = 128 * 256, b1n = 128, w2single = 32 * 128, b2n = 32;
  uint64_t rng = 0x12345678ULL;
  auto nextU = [&]() {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(rng >> 11) / static_cast<double>(0x1FFFFFFFFFFFFFULL);
  };
  for (size_t i = 0; i < w1n; ++i) flatSingle[i] = static_cast<float>((nextU() - 0.5) * 0.1);
  for (size_t i = 0; i < b1n; ++i) flatSingle[w1n + i] = static_cast<float>((nextU() - 0.5) * 0.02);
  for (size_t i = 0; i < w2single; ++i)
    flatSingle[w1n + b1n + i] = static_cast<float>((nextU() - 0.5) * 0.1);
  for (size_t i = 0; i < b2n; ++i)
    flatSingle[w1n + b1n + w2single + i] = static_cast<float>((nextU() - 0.5) * 0.02);
  GroupedMlp mlpB;
  CHECK(mlpB.setTensors(names, flatSingle));
  std::vector<float> flatPair;
  flatPair.insert(flatPair.end(), flatSingle.begin(), flatSingle.begin() + w1n + b1n);
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 128; ++c) flatPair.push_back(flatSingle[w1n + b1n + r * 128 + c]);
    for (int c = 0; c < 128; ++c) flatPair.push_back(0.5f * flatSingle[w1n + b1n + r * 128 + c]);
  }
  flatPair.insert(flatPair.end(), flatSingle.begin() + w1n + b1n + w2single, flatSingle.end());
  GroupedMlp mlpPair;
  CHECK(mlpPair.setTensors(names, flatPair));
  CHECK(mlpPair.parameterCount() == flatPair.size());
  CHECK(mlpPair.parameterCount() == mlpB.parameterCount() + 32 * 128);
  float tok[256];
  for (int i = 0; i < 256; ++i) tok[i] = static_cast<float>((i % 41)) / 40.0f;
  float vS, vP, wS[3], wP[3];
  mlpB.forward(tok, vS, wS);
  mlpPair.forward(tok, vP, wP);
  CHECK(std::isfinite(vP));
  bool differs = (std::fabs(vS - vP) > 1e-6);
  for (int i = 0; i < 3; ++i) {
    CHECK(std::isfinite(wP[i]));
    if (std::fabs(wS[i] - wP[i]) > 1e-6) differs = true;
  }
  CHECK(differs);
  std::vector<std::string> n2;
  std::vector<std::vector<int>> s2;
  std::vector<const float*> d2;
  mlpPair.getTensors(n2, s2, d2);
  CHECK(s2[2][0] == 32 && s2[2][1] == 256);
  std::vector<float> flat2;
  for (size_t i = 0; i < d2.size(); ++i) {
    size_t n = 1;
    for (int s : s2[i]) n *= static_cast<size_t>(s);
    for (size_t j = 0; j < n; ++j) flat2.push_back(d2[i][j]);
  }
  GroupedMlp mlpPair2;
  CHECK(mlpPair2.setTensors(n2, flat2));
  float vP2, wP2[3];
  mlpPair2.forward(tok, vP2, wP2);
  CHECK_CLOSE(vP, vP2, 1e-7);
  for (int i = 0; i < 3; ++i) CHECK_CLOSE(wP[i], wP2[i], 1e-7);
}

void testQuantization() {
  EmbeddingTables tables;
  tables.init(7);
  QuantEmbeddingTables qt;
  QuantScales scales;
  qt.quantizeFrom(tables, scales);
  EmbeddingTables back;
  qt.dequantizeTo(back, scales);
  double maxErr = 0.0;
  size_t n = 0;
  for (int g = 0; g < 8; ++g) {
    const std::vector<float>& a = tables.groupData(g);
    const std::vector<float>& b2 = back.groupData(g);
    for (size_t i = 0; i < a.size(); ++i) {
      double e = std::fabs(a[i] - b2[i]);
      if (e > maxErr) maxErr = e;
      ++n;
    }
  }
  CHECK(n > 0);
  CHECK(maxErr < 0.002);
  Board board;
  std::vector<ActiveFeature> feats;
  GroupedFeatureSet::extract(board, feats);
  GroupedAccumulator accF(&tables);
  accF.refresh(feats);
  GroupedAccumulatorInt accI;
  accI.bind(&qt, &scales);
  accI.refresh(feats);
  float tf[256], ti[256];
  accF.tokens(tf);
  accI.tokens(ti);
  double worst = 0.0;
  for (int i = 0; i < 256; ++i) {
    double e = std::fabs(tf[i] - ti[i]);
    if (e > worst) worst = e;
  }
  CHECK(worst < 0.05);
}
