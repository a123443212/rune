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

#include "core/accumulators/flex_accumulator.h"
#include "core/accumulators/token_layout.h"
#include "core/architectures/relational/relational.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"
#include "core/model_io/model_io.h"
#include "tests/cpp/test_framework.h"

using namespace rune;

static float gateRef(const std::string& gate, float s) {
  if (gate == "hard_sigmoid") {
    float v = 0.2f * s + 0.5f;
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
  }
  if (s < 0.0f) return 0.0f;
  if (s > 1.0f) return 1.0f;
  return s;
}

void testRelationalMath() {
  for (const char* gate : {"clip", "hard_sigmoid"}) {
    for (bool dyn : {false, true}) {
      RelationalModel m;
      RelationalConfig cfg;
      cfg.tokens = 8;
      cfg.dim = 32;
      gateFromString(gate, cfg.gate);
      cfg.alpha = 0.5f;
      cfg.dynamicBias = dyn;
      std::string err;
      CHECK(m.configure(cfg, err));
      for (float& u : m.mixer.dynU) u = 0.1f;
      for (float& w : m.mixer.dynW) w = -0.05f;
      float x[256];
      for (int i = 0; i < 256; ++i) x[i] = static_cast<float>((i % 13) - 6) * 0.04f;
      float ctx[12] = {1.0f, 0.5f, 0.25f, 0.5f, 0.25f, 0.0f, 0.5f, 0.75f, 1.0f, 0.0f, 0.0f, 0.0f};
      float out[256];
      m.mixer.forward(x, ctx, out);
      float q[256], k[256], v[256];
      for (int t = 0; t < 8; ++t) {
        simd::matVec(m.mixer.wq.data(), x + t * 32, m.mixer.bq.data(), q + t * 32, 32, 32);
        simd::matVec(m.mixer.wk.data(), x + t * 32, m.mixer.bk.data(), k + t * 32, 32, 32);
        simd::matVec(m.mixer.wv.data(), x + t * 32, m.mixer.bv.data(), v + t * 32, 32, 32);
      }
      for (int a = 0; a < 8; ++a) {
        for (int d = 0; d < 32; ++d) {
          float expect = x[a * 32 + d];
          for (int b = 0; b < 8; ++b) {
            float s = m.mixer.gabS[a * 8 + b];
            for (int dd = 0; dd < 32; ++dd) s += q[a * 32 + dd] * k[b * 32 + dd];
            if (dyn) {
              float u = 0.0f;
              float w = 0.0f;
              for (int c = 0; c < ContextSpec::kDim; ++c) {
                u += m.mixer.dynU[a * 8 + c] * ctx[c];
                w += m.mixer.dynW[b * 8 + c] * ctx[c];
              }
              float delta = u * w;
              if (delta < -0.25f) delta = -0.25f;
              if (delta > 0.25f) delta = 0.25f;
              s += delta;
            }
            expect += 0.5f * gateRef(gate, s) * v[b * 32 + d];
          }
          CHECK_CLOSE(out[a * 32 + d], expect, 1e-4);
        }
      }
    }
  }
}

void testDynamicBiasBounded() {
  RelationalModel m;
  RelationalConfig cfg;
  cfg.dynamicBias = true;
  std::string err;
  CHECK(m.configure(cfg, err));
  for (float& u : m.mixer.dynU) u = 10.0f;
  for (float& w : m.mixer.dynW) w = 10.0f;
  float x[256] = {0.0f};
  for (int i = 0; i < 256; ++i) x[i] = 0.1f;
  float ctx[17] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float out[256];
  m.mixer.forward(x, ctx, out);
  for (int i = 0; i < 256; ++i) CHECK(std::isfinite(out[i]));
  RelationalModel ms;
  RelationalConfig cfgs;
  CHECK(ms.configure(cfgs, err));
  ms.mixer.wq = m.mixer.wq;
  float outS[256];
  float zeroCtx[17] = {0};
  ms.mixer.forward(x, zeroCtx, outS);
  bool differs = false;
  for (int i = 0; i < 256; ++i) {
    if (std::fabs(out[i] - outS[i]) > 1e-6) differs = true;
  }
  CHECK(differs);
}

void testContextValues() {
  Board b;
  float ctx[17];
  computeContext(b, ctx);
  CHECK_CLOSE(ctx[0], 0.0, 1e-6);
  CHECK_CLOSE(ctx[1], 0.0, 1e-6);
  CHECK_CLOSE(ctx[2], 1.0, 1e-6);
  CHECK_CLOSE(ctx[3], 1.0, 1e-6);
  CHECK_CLOSE(ctx[4], 0.5, 1e-6);
  CHECK_CLOSE(ctx[5], 0.5, 1e-6);
  CHECK_CLOSE(ctx[6], 0.5, 1e-6);
  CHECK_CLOSE(ctx[7], 0.5, 1e-6);
  CHECK_CLOSE(ctx[8], 0.5, 1e-6);
  CHECK_CLOSE(ctx[9], 0.5, 1e-6);
  CHECK_CLOSE(ctx[10], 1.0, 1e-6);
  CHECK_CLOSE(ctx[11], 1.0, 1e-6);
  CHECK_CLOSE(ctx[12], 0.625, 1e-6);
  CHECK_CLOSE(ctx[13], 1.0, 1e-6);
  CHECK_CLOSE(ctx[14], 0.0, 1e-6);
  CHECK_CLOSE(ctx[15], 0.0, 1e-6);
  CHECK_CLOSE(ctx[16], 0.0, 1e-6);
  Board end("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
  computeContext(end, ctx);
  CHECK_CLOSE(ctx[1], 1.0, 1e-6);
  CHECK_CLOSE(ctx[10], 0.3125, 1e-6);
  CHECK_CLOSE(ctx[11], 0.3125, 1e-6);
}

void testTokenLayouts() {
  std::string err;
  TokenLayout l8, l6, l10;
  CHECK(TokenLayout::make(8, 32, l8, err));
  CHECK(TokenLayout::make(6, 32, l6, err));
  CHECK(TokenLayout::make(10, 40, l10, err));
  TokenLayout bad;
  CHECK(!TokenLayout::make(7, 32, bad, err));
  CHECK(!TokenLayout::make(8, 16, bad, err));
  for (int g = 0; g < 8; ++g) CHECK(l8.findToken(g, 0) == g);
  CHECK(l6.findToken(3, 10) == l6.findToken(4, 20));
  CHECK(l6.findToken(5, 0) == l6.findToken(6, 0));
  CHECK(l6.findToken(3, 0) != l6.findToken(5, 0));
  CHECK(l10.findToken(2, 10) != l10.findToken(2, 200));
  CHECK(l10.findToken(5, 10) != l10.findToken(5, 400));
  FlexEmbeddings emb;
  emb.configure(32);
  emb.init(5);
  FlexAccumulator inc;
  FlexAccumulator ref;
  inc.configure(&emb, &l6);
  ref.configure(&emb, &l6);
  Board b;
  std::vector<ActiveFeature> prev;
  GroupedFeatureSet::extract(b, prev);
  inc.refresh(prev);
  std::vector<Move> moves;
  b.generateLegalMoves(moves);
  CHECK(b.makeMove(moves[3]));
  std::vector<ActiveFeature> cur;
  GroupedFeatureSet::extract(b, cur);
  std::vector<ActiveFeature> added, removed;
  GroupedFeatureSet::diffFeatures(prev, cur, added, removed);
  inc.applyDiff(added, removed);
  ref.refresh(cur);
  float ti[192], tr[192];
  inc.tokens(ti);
  ref.tokens(tr);
  for (int i = 0; i < 192; ++i) CHECK_CLOSE(ti[i], tr[i], 1e-5);
  b.unmakeMove();
}

void testFlexModelIO() {
  for (const char* quant : {"fp32", "int8", "int16"}) {
    TokenLayout layout;
    std::string err;
    CHECK(TokenLayout::make(8, 32, layout, err));
    FlexEmbeddings emb;
    emb.configure(32);
    emb.init(9);
    RelationalModel m;
    RelationalConfig cfg;
    cfg.dynamicBias = true;
    CHECK(m.configure(cfg, err));
    ModelSpec spec = m.spec();
    std::string path = std::string("/tmp/rune_flex_") + quant + ".rune";
    CHECK(saveFlexRuneFile(path, spec, emb, m, quant, err));
    RuneFile loaded;
    CHECK(loadRuneFile(path, loaded, err));
    CHECK(loaded.isFlex);
    CHECK(loaded.spec.tokens == 8 && loaded.spec.tokenDim == 32);
    CHECK(loaded.spec.gate == "clip");
    Board b;
    RelationalEvaluator ref;
    CHECK(ref.configure(&emb, &layout, &m, err));
    RelationalEvaluator got;
    CHECK(got.configure(&loaded.flexEmbeddings, &loaded.layout,
                        static_cast<RelationalModel*>(loaded.arch.get()), err));
    EvalResultFlex r1 = ref.evaluateBoard(b);
    EvalResultFlex r2 = got.evaluateBoard(b);
    float tol = (std::string(quant) == "fp32") ? 1e-6f : 0.02f;
    CHECK_CLOSE(r1.value, r2.value, tol);
  }
}

void testInt16FixedPath() {
  EmbeddingTables tables;
  tables.init(13);
  Quant16Tables q16;
  QuantScales sc;
  q16.quantizeFrom(tables, sc);
  GroupedAccumulator ref(&tables);
  GroupedAccumulator16 acc;
  acc.bind(&q16, &sc);
  Board b("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
  std::vector<ActiveFeature> feats;
  GroupedFeatureSet::extract(b, feats);
  ref.refresh(feats);
  acc.refresh(feats);
  float tf[256], ti[256];
  ref.tokens(tf);
  acc.tokens(ti);
  double worst = 0.0;
  for (int i = 0; i < 256; ++i) {
    double e = std::fabs(tf[i] - ti[i]);
    if (e > worst) worst = e;
  }
  CHECK(worst < 0.001);
}
