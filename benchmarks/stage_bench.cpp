#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/accumulators/grouped_accumulator.h"
#include "core/architectures/attention/attention.h"
#include "core/architectures/mlp/mlp.h"
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
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--arch") == 0) arch = argv[i + 1];
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
  std::printf("arch: %s\n", arch.c_str());
  std::printf("mlp_params: %zu\n", mlp.parameterCount());
  std::printf("attn_params: %zu\n", attnGab.parameterCount());
  return 0;
}
