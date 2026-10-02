#include <chrono>
#include <cstdio>
#include <random>
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
static double benchMs(F&& f, int iters) {
  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) f(i);
  auto t1 = Clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
  Board start;
  std::vector<ActiveFeature> feats;
  double tFeat = benchMs(
      [&](int) {
        GroupedFeatureSet::extract(start, feats);
      },
      2000);
  std::printf("feature_extract_us: %.2f\n", tFeat * 1000.0 / 2000);

  EmbeddingTables tables;
  tables.init(3);
  GroupedAccumulator acc(&tables);
  GroupedFeatureSet::extract(start, feats);
  double tRefresh = benchMs([&](int) { acc.refresh(feats); }, 5000);
  std::printf("accumulator_refresh_us: %.2f\n", tRefresh * 1000.0 / 5000);

  std::vector<Move> moves;
  start.generateLegalMoves(moves);
  Board moved = start;
  moved.makeMove(moves[0]);
  std::vector<ActiveFeature> f2;
  GroupedFeatureSet::extract(moved, f2);
  std::vector<ActiveFeature> added, removed;
  GroupedFeatureSet::diffFeatures(feats, f2, added, removed);
  double tUpdate = benchMs([&](int) { acc.applyDiff(added, removed); }, 20000);
  std::printf("accumulator_update_us: %.3f\n", tUpdate * 1000.0 / 20000);

  float tok[256];
  acc.tokens(tok);
  RuneAttnModel attn(true);
  float out[256];
  double tAttn = benchMs([&](int) { attn.attn.forward(tok, out); }, 2000);
  std::printf("attention_us: %.2f\n", tAttn * 1000.0 / 2000);

  GroupedMlp mlp;
  float v;
  float wdl[3];
  double tHead = benchMs([&](int) { mlp.forward(tok, v, wdl); }, 5000);
  std::printf("head_mlp_us: %.2f\n", tHead * 1000.0 / 5000);

  Evaluator ev(&tables, &attn);
  double tFull = benchMs(
      [&](int i) {
        (void)i;
        ev.evaluateBoard(start);
      },
      2000);
  std::printf("full_eval_refresh_us: %.2f\n", tFull * 1000.0 / 2000);

  ev.refresh(start);
  double tInc = benchMs([&](int) { ev.evaluate(); }, 5000);
  std::printf("full_eval_incremental_us: %.2f\n", tInc * 1000.0 / 5000);

  std::mt19937 rng(42);
  const int kNodes = 20000;
  Board b;
  auto t0 = Clock::now();
  int evals = 0;
  Evaluator evb(&tables, &mlp);
  evb.refresh(b);
  std::vector<ActiveFeature> prev;
  GroupedFeatureSet::extract(b, prev);
  for (int i = 0; i < kNodes; ++i) {
    std::vector<Move> lm;
    b.generateLegalMoves(lm);
    if (lm.empty()) {
      b = Board();
      evb.refresh(b);
      GroupedFeatureSet::extract(b, prev);
      continue;
    }
    Move m = lm[rng() % lm.size()];
    b.makeMove(m);
    std::vector<ActiveFeature> cur;
    GroupedFeatureSet::extract(b, cur);
    std::vector<ActiveFeature> ad, rm;
    GroupedFeatureSet::diffFeatures(prev, cur, ad, rm);
    evb.updateIncremental(ad, rm);
    volatile EvalResult r = evb.evaluate();
    (void)r;
    ++evals;
    prev = cur;
  }
  auto t1 = Clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::printf("walk_nodes: %d walk_ms: %.1f nps: %.0f\n", evals, ms, evals * 1000.0 / ms);
  std::printf("mlp_params: %zu attn_params: %zu\n", mlp.parameterCount(), attn.parameterCount());
  return 0;
}
