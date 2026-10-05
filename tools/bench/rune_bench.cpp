#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include "core/accumulators/grouped_accumulator.h"
#include "core/architectures/attention/attention.h"
#include "core/architectures/mlp/mlp.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"
#include "core/inference/evaluator.h"
#include "core/kernels/simd_kernels.h"
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
  int threads = 1;
  std::string positions;
  std::string path = "auto";
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--threads") == 0) threads = std::atoi(argv[i + 1]);
    if (std::strcmp(argv[i], "--positions") == 0) positions = argv[i + 1];
    if (std::strcmp(argv[i], "--path") == 0) path = argv[i + 1];
  }
  if (path == "scalar") kern::setPathForTest(kern::Path::Scalar);
  if (path == "avx2") kern::setPathForTest(kern::Path::Avx2);
  (void)positions;
  Board start;
  std::vector<ActiveFeature> f0;
  GroupedFeatureSet::extract(start, f0);
  EmbeddingTables tables;
  tables.init(3);
  GroupedAccumulator acc(&tables);
  acc.refresh(f0);
  std::vector<Move> lm;
  start.generateLegalMoves(lm);
  Board moved = start;
  moved.makeMove(lm[0]);
  std::vector<ActiveFeature> f1;
  GroupedFeatureSet::extract(moved, f1);
  std::vector<ActiveFeature> added, removed;
  GroupedFeatureSet::diffFeatures(f0, f1, added, removed);
  float tok[256];
  double uFeat = benchUs([&](int) {
    std::vector<ActiveFeature> tmp;
    GroupedFeatureSet::extract(start, tmp);
  }, 2000);
  double uRefresh = benchUs([&](int) { acc.refresh(f0); }, 5000);
  double uUpdate = benchUs([&](int) { acc.applyDiff(added, removed); }, 20000);
  double uTokens = benchUs([&](int) { acc.tokens(tok); }, 20000);
  RuneAttnModel attn(true);
  GroupedMlp mlp;
  float out[256];
  double uMixer = benchUs([&](int) { attn.attn.forward(tok, out); }, 2000);
  float v;
  float wdl[3];
  double uHead = benchUs([&](int) { mlp.forward(tok, v, wdl); }, 5000);
  Evaluator ev(&tables, &attn);
  double uFull = benchUs([&](int) { ev.evaluateBoard(start); }, 2000);
  ev.refresh(start);
  double uInc = benchUs([&](int) { ev.evaluate(); }, 5000);
  std::printf("path %s threads %d\n", kern::activePathName(), threads);
  std::printf("feature_extract_us %.3f\n", uFeat);
  std::printf("accumulator_refresh_us %.3f\n", uRefresh);
  std::printf("accumulator_update_us %.3f\n", uUpdate);
  std::printf("tokenization_us %.3f\n", uTokens);
  std::printf("mixer_us %.3f\n", uMixer);
  std::printf("head_us %.3f\n", uHead);
  std::printf("full_eval_refresh_us %.3f\n", uFull);
  std::printf("full_eval_incremental_us %.3f\n", uInc);
  std::printf("eval_per_sec_refresh %.0f\n", 1000000.0 / uFull);
  std::printf("eval_per_sec_incremental %.0f\n", 1000000.0 / uInc);
  if (threads > 1) {
    auto t0 = Clock::now();
    std::vector<std::thread> th;
    int per = 2000 / threads;
    for (int t = 0; t < threads; ++t) {
      th.emplace_back([&, t]() {
        Evaluator e(&tables, &attn);
        for (int i = 0; i < per; ++i) e.evaluateBoard(start);
      });
    }
    for (auto& x : th) x.join();
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("mt_threads %d mt_ms %.2f mt_nps %.0f\n", threads, ms, 2000 * 1000.0 / ms);
  }
  return 0;
}
