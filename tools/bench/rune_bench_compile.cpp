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
#include "core/kernels/fused.h"
#include "core/kernels/simd_kernels.h"
#include "core/runtime/arena.h"
using namespace rune;
using Clock = std::chrono::high_resolution_clock;
template <typename F>
static double benchUs(F&& f, int iters) {
  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) f(i);
  auto t1 = Clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / (double)iters;
}
int main(int argc, char** argv) {
  int iters = 2000;
  std::string mode = "all";
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--iters") == 0) iters = std::atoi(argv[i + 1]);
    if (std::strcmp(argv[i], "--mode") == 0) mode = argv[i + 1];
  }
  Board start;
  std::vector<ActiveFeature> f0;
  GroupedFeatureSet::extract(start, f0);
  EmbeddingTables tables;
  tables.init(3);
  GroupedAccumulator acc(&tables);
  acc.refresh(f0);
  float tok[256];
  acc.tokens(tok);
  float w[1024];
  float b[32];
  for (int i = 0; i < 1024; ++i) w[i] = 0.02f;
  for (int i = 0; i < 32; ++i) b[i] = 0.01f;
  float q[256];
  float k[256];
  float v[256];
  float s[64];
  float g[64];
  float gab[64];
  for (int i = 0; i < 64; ++i) gab[i] = 0.0f;
  float tmp[256];
  float out[256];
  double uQkv = benchUs([&](int) { fused::qkvFused8x32(w, b, w, b, w, b, tok, q, k, v); }, iters);
  double uScore = benchUs([&](int) { fused::scoreBiasGate8x8(q, k, gab, GateFn::Clip, s, g); }, iters);
  double uMix = benchUs([&](int) { fused::mixResidual8x32(g, v, tok, 1.0f, tmp, out); }, iters);
  RuneAttnModel attn(true);
  GroupedMlp mlp;
  float mo[256];
  double uMixerG = benchUs([&](int) { attn.attn.forward(tok, mo); }, iters);
  float vv;
  float wdl[3];
  double uHeadG = benchUs([&](int) { mlp.forward(tok, vv, wdl); }, iters);
  Evaluator ev(&tables, &attn);
  double uFull = benchUs([&](int) { ev.evaluateBoard(start); }, iters);
  rt::Arena arena(rt::arenaBytesFor(8, 32, 128, 32));
  std::printf("iters %d\n", iters);
  std::printf("qkv_fused_us %.3f\n", uQkv);
  std::printf("score_bias_gate_us %.3f\n", uScore);
  std::printf("mix_residual_us %.3f\n", uMix);
  std::printf("mixer_generic_us %.3f\n", uMixerG);
  std::printf("head_generic_us %.3f\n", uHeadG);
  std::printf("full_eval_us %.3f\n", uFull);
  std::printf("arena_bytes %zu\n", arena.bytes());
  std::printf("path %s\n", kern::activePathName());
  (void)mode;
  return 0;
}
