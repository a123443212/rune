#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/accumulators/flex_accumulator.h"
#include "core/accumulators/grouped_accumulator.h"
#include "core/accumulators/token_layout.h"
#include "core/architectures/attention/attention.h"
#include "core/architectures/dense/dense.h"
#include "core/architectures/mlp/mlp.h"
#include "core/architectures/relational/relational.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"
#include "core/inference/evaluator.h"

using namespace rune;
using Clock = std::chrono::high_resolution_clock;

template <typename F>
static double benchUs(F&& f, int iters) {
  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) f(i);
  auto t1 = Clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

static void report(const char* key, double us) { std::printf("%s: %.3f\n", key, us); }

int main(int argc, char** argv) {
  std::string arch = "RUNE-ATTN-GAB";
  int flexTokens = 8;
  int flexDim = 32;
  bool flexDynamic = true;
  std::string denseDims;
  std::string densePool = "none";
  bool denseGate = false;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--arch") == 0) arch = argv[i + 1];
    if (std::strcmp(argv[i], "--tokens") == 0) flexTokens = std::atoi(argv[i + 1]);
    if (std::strcmp(argv[i], "--dim") == 0) flexDim = std::atoi(argv[i + 1]);
    if (std::strcmp(argv[i], "--bias") == 0) flexDynamic = std::string(argv[i + 1]) == "dynamic";
    if (std::strcmp(argv[i], "--dims") == 0) denseDims = argv[i + 1];
    if (std::strcmp(argv[i], "--pool") == 0) densePool = argv[i + 1];
    if (std::strcmp(argv[i], "--gate") == 0) denseGate = std::string(argv[i + 1]) == "on";
  }

  Board start;
  Board moved;
  {
    std::vector<Move> lm;
    start.generateLegalMoves(lm);
    moved = start;
    moved.makeMove(lm[0]);
  }
  std::vector<ActiveFeature> f0, f1, added, removed;
  GroupedFeatureSet::extract(start, f0);
  GroupedFeatureSet::extract(moved, f1);

  report("feature_extract_us", benchUs([&](int) {
           std::vector<ActiveFeature> tmp;
           GroupedFeatureSet::extract(start, tmp);
         }, 2000));
  report("feature_update_us", benchUs([&](int) {
           std::vector<ActiveFeature> a, r;
           GroupedFeatureSet::diffFeatures(f0, f1, a, r);
         }, 20000));

  EmbeddingTables tables;
  tables.init(3);
  GroupedAccumulator acc(&tables);
  acc.refresh(f0);
  GroupedFeatureSet::diffFeatures(f0, f1, added, removed);
  report("accumulator_refresh_us", benchUs([&](int) { acc.refresh(f0); }, 5000));
  report("accumulator_update_us",
         benchUs([&](int) { acc.applyDiff(added, removed); }, 20000));
  float tok[256];
  report("token_create_us", benchUs([&](int) { acc.tokens(tok); }, 20000));

  RuneAttentionBlock gab(true);
  float q[256], k[256], v[256], s[64], y[256], mix[256];
  auto doQkv = [&]() {
    for (int t = 0; t < 8; ++t) {
      simd::matVec(gab.wq.data(), tok + t * 32, gab.bq.data(), q + t * 32, 32, 32);
      simd::matVec(gab.wk.data(), tok + t * 32, gab.bk.data(), k + t * 32, 32, 32);
      simd::matVec(gab.wv.data(), tok + t * 32, gab.bv.data(), v + t * 32, 32, 32);
    }
  };
  doQkv();
  report("qkv_proj_us", benchUs([&](int) { doQkv(); }, 5000));
  report("qkt_us", benchUs([&](int) { simd::matMulTT(q, k, s, 8, 8, 32); }, 10000));
  report("gab_bias_act_us", benchUs([&](int) {
            simd::matMulTT(q, k, s, 8, 8, 32);
            for (int i = 0; i < 64; ++i) s[i] = simd::clippedRelu(s[i] + gab.gab[i]);
          }, 5000));
  report("v_mix_us", benchUs([&](int) { simd::matMul(s, v, y, 8, 32, 8); }, 10000));
  report("residual_us", benchUs([&](int) {
            for (int i = 0; i < 256; ++i) mix[i] = tok[i] + y[i];
          }, 50000));

  GroupedMlp mlp;
  RuneAttnModel attn(false);
  RuneAttnModel attnGab(true);
  float value;
  float wdl[3];
  report("head_mlp_us", benchUs([&](int) { mlp.forward(tok, value, wdl); }, 5000));

  Evaluator evMlp(&tables, &mlp);
  Evaluator evAttn(&tables, &attn);
  Evaluator evGab(&tables, &attnGab);
  Evaluator* ev = &evGab;
  if (arch == "RUNE-MLP") ev = &evMlp;
  if (arch == "RUNE-ATTN") ev = &evAttn;
  report("full_eval_refresh_us", benchUs([&](int) { ev->evaluateBoard(start); }, 2000));
  ev->refresh(start);
  report("full_eval_incremental_us", benchUs([&](int) { ev->evaluate(); }, 5000));

  TokenLayout layout;
  std::string layoutErr;
  if (TokenLayout::make(flexTokens, flexDim, layout, layoutErr)) {
    FlexEmbeddings femb;
    femb.configure(flexDim);
    femb.init(3);
    RelationalModel rel;
    RelationalConfig rcfg;
    rcfg.tokens = flexTokens;
    rcfg.dim = flexDim;
    rcfg.dynamicBias = flexDynamic;
    std::string rcerr;
    if (rel.configure(rcfg, rcerr)) {
      RelationalEvaluator relev;
      std::string reerr;
      if (relev.configure(&femb, &layout, &rel, reerr)) {
        report("rel_full_eval_refresh_us",
               benchUs([&](int) { relev.evaluateBoard(start); }, 2000));
        relev.refresh(start);
        report("rel_full_eval_incremental_us", benchUs([&](int) { relev.evaluate(); }, 5000));
        std::vector<float> rt(flexTokens * flexDim);
        relev.currentTokens(rt.data());
        float ctx[8];
        relev.currentContext(ctx);
        std::vector<float> rout(flexTokens * flexDim);
        report("rel_mixer_us",
               benchUs([&](int) { rel.mixer.forward(rt.data(), ctx, rout.data()); }, 3000));
        float rv;
        float rwdl[3];
        report("rel_head_us",
               benchUs([&](int) { rel.forwardWithContext(rt.data(), ctx, rv, rwdl); }, 3000));
      }
    }
    std::printf("rel_params: %zu\n", rel.parameterCount() + femb.numFloats());
    std::printf("rel_tokens: %d\nrel_dim: %d\nrel_dynamic: %d\n", flexTokens, flexDim,
                flexDynamic ? 1 : 0);
  }
  std::printf("arch: %s\n", arch.c_str());
  std::printf("mlp_params: %zu\n", mlp.parameterCount());
  std::printf("attn_params: %zu\n", attnGab.parameterCount());

  if (!denseDims.empty()) {
    std::vector<int> dims;
    std::string cur;
    for (char c : denseDims + ",") {
      if (c == ',') {
        if (!cur.empty()) dims.push_back(std::atoi(cur.c_str()));
        cur.clear();
      } else {
        cur += c;
      }
    }
    DenseBuildSpec dspec;
    dspec.variant = "B";
    dspec.dims = dims;
    dspec.pooling = densePool;
    dspec.poolClip = true;
    dspec.gateOn = denseGate;
    dspec.sharedWidth = 32;
    DenseModel dmodel;
    std::string derr;
    if (dmodel.configure(dspec, derr)) {
      VarWidths gw;
      for (int g = 0; g < 8; ++g) gw.w[g] = (densePool == "shared") ? 32 : dims[g];
      VarEmbeddings vemb;
      vemb.configure(gw);
      vemb.init(3);
      DenseEvaluator dev;
      std::string deerr;
      if (dev.configure(&vemb, &dmodel, deerr)) {
        {
          float dv;
          float dwdl[3];
          report("dense_full_eval_refresh_us", benchUs([&](int) {
                   dev.refresh(start);
                   dev.evaluate(dv, dwdl);
                 }, 2000));
        }
        {
          float dv;
          float dwdl[3];
          report("dense_full_eval_incremental_us", benchUs([&](int) {
                   dev.evaluate(dv, dwdl);
                 }, 5000));
        }
        int inTotal = 0;
        for (int g = 0; g < 8; ++g) inTotal += gw.w[g];
        int outTotal = 0;
        for (int d : dims) outTotal += d;
        std::vector<float> raw(inTotal), formed(outTotal), gated(outTotal);
        dev.currentTokens(raw.data());
        report("dense_token_projection_us", benchUs([&](int) {
                 dmodel.pool.forward(raw.data(), formed.data());
               }, 5000));
        dmodel.pool.forward(raw.data(), formed.data());
        report("dense_gating_us", benchUs([&](int) {
                 dmodel.gate.forward(formed.data(), gated.data());
               }, 20000));
        float dv;
        float dwdl[3];
        report("dense_model_forward_us",
               benchUs([&](int) { dmodel.forward(raw.data(), dv, dwdl); }, 3000));
        std::printf("dense_params: %zu\n", dmodel.parameterCount() + vemb.numFloats());
        std::printf("dense_dims_total: %d\n", outTotal);
      } else {
        std::printf("dense_eval_error: %s\n", deerr.c_str());
      }
    } else {
      std::printf("dense_config_error: %s\n", derr.c_str());
    }
  }
  return 0;
}
