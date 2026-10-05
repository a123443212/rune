#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "core/architectures/attention/attention.h"
#include "core/architectures/mlp/mlp.h"
#include "core/accumulators/grouped_accumulator.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"
#include "core/inference/evaluator.h"
#include "core/inference/dump.h"
#include "core/kernels/ref_kernels.h"
#include "core/kernels/simd_kernels.h"
using namespace rune;
using Clock = std::chrono::high_resolution_clock;
static double maxAbsDiff(const float* a, const float* b, int n, int* argmax) {
  double m = 0.0;
  int ai = 0;
  for (int i = 0; i < n; ++i) {
    double d = (double)a[i] - (double)b[i];
    if (d < 0) d = -d;
    if (d > m) { m = d; ai = i; }
  }
  if (argmax) *argmax = ai;
  return m;
}
int main(int argc, char** argv) {
  std::string positions;
  std::string dumpDir;
  std::string mode = "exact";
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--positions") == 0) positions = argv[i + 1];
    if (std::strcmp(argv[i], "--dump-dir") == 0) dumpDir = argv[i + 1];
    if (std::strcmp(argv[i], "--mode") == 0) mode = argv[i + 1];
  }
  std::vector<std::string> fens;
  fens.push_back("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  fens.push_back("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
  fens.push_back("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1");
  if (!positions.empty()) {
    std::ifstream pf(positions);
    if (pf) {
      fens.clear();
      std::string line;
      while (std::getline(pf, line)) {
        if (!line.empty()) fens.push_back(line);
      }
    }
  }
  EmbeddingTables tables;
  tables.init(7);
  RuneAttnModel attn(true);
  Evaluator ev(&tables, &attn);
  double globalMax = 0.0;
  for (size_t fi = 0; fi < fens.size(); ++fi) {
    Board b;
    if (!b.setFen(fens[fi])) {
      std::printf("pos %zu bad fen\n", fi);
      return 1;
    }
    std::vector<ActiveFeature> feats;
    GroupedFeatureSet::extract(b, feats);
    GroupedAccumulator acc(&tables);
    acc.refresh(feats);
    float tokRef[256];
    acc.tokens(tokRef);
    float tokSimd[256];
    acc.tokens(tokSimd);
    int am = 0;
    double d = maxAbsDiff(tokRef, tokSimd, 256, &am);
    if (d > globalMax) globalMax = d;
    ev.refresh(b);
    EvalResult r = ev.evaluate();
    std::vector<ActiveFeature> feats2 = feats;
    GroupedAccumulator acc2(&tables);
    acc2.refresh(feats2);
    float t2[256];
    acc2.tokens(t2);
    int am2 = 0;
    double d2 = maxAbsDiff(tokRef, t2, 256, &am2);
    if (d2 > globalMax) globalMax = d2;
    if (!dumpDir.empty()) {
      IntermediateDump dump;
      dump.features = feats;
      dump.accumulator.assign(256, 0.0f);
      dump.tokens.assign(tokRef, tokRef + 256);
      dump.value = r.value;
      dump.wdl[0] = r.wdl[0];
      dump.wdl[1] = r.wdl[1];
      dump.wdl[2] = r.wdl[2];
      char path[512];
      std::snprintf(path, sizeof(path), "%s/pos_%03zu.txt", dumpDir.c_str(), fi);
      dump.writeText(path);
    }
    std::printf("pos %zu feats %zu value %.6f maxdiff %.9g\n", fi, feats.size(), (double)r.value, d);
  }
  double tol = (mode == "exact") ? 0.0 : 1e-5;
  std::printf("global_max_abs_diff %.9g tol %.9g %s\n", globalMax, tol, globalMax <= tol ? "PASS" : "FAIL");
  return globalMax <= tol ? 0 : 2;
}
