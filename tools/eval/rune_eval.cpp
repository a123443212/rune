#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "core/board/board.h"
#include "core/inference/evaluator.h"
#include "core/model_io/model_io.h"
#include "core/architectures/adaptive/adaptive.h"
#include "core/architectures/dense/dense.h"
#include "core/architectures/relational/relational.h"
#include "core/kernels/simd_kernels.h"
int main(int argc, char** argv) {
  std::string model;
  std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  std::string path = "auto";
  float thresholdOverride = 0.0f;
  bool useThreshold = false;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--model") == 0) model = argv[i + 1];
    if (std::strcmp(argv[i], "--fen") == 0) fen = argv[i + 1];
    if (std::strcmp(argv[i], "--path") == 0) path = argv[i + 1];
    if (std::strcmp(argv[i], "--threshold") == 0) {
      thresholdOverride = static_cast<float>(std::atof(argv[i + 1]));
      useThreshold = true;
    }
  }
  if (path == "scalar") rune::kern::setPathForTest(rune::kern::Path::Scalar);
  if (path == "avx2") rune::kern::setPathForTest(rune::kern::Path::Avx2);
  if (model.empty()) {
    std::printf("need --model\n");
    return 1;
  }
  rune::RuneFile rf;
  std::string err;
  if (!rune::loadRuneFile(model, rf, err)) {
    std::printf("load fail %s\n", err.c_str());
    return 1;
  }
  rune::Board b;
  if (!b.setFen(fen)) {
    std::printf("bad fen\n");
    return 1;
  }
  if (rf.isFlex) {
    rune::RelationalEvaluator rev;
    std::string e2;
    if (!rev.configure(&rf.flexEmbeddings, &rf.layout, static_cast<rune::RelationalModel*>(rf.arch.get()), e2)) {
      std::printf("configure fail %s\n", e2.c_str());
      return 1;
    }
    auto r = rev.evaluateBoard(b);
    std::printf("%.9g %.9g %.9g %.9g\n", (double)r.value, (double)r.wdl[0], (double)r.wdl[1], (double)r.wdl[2]);
    return 0;
  }
  if (rf.isDense) {
    rune::DenseEvaluator dev;
    std::string e2;
    if (!dev.configure(&rf.varEmbeddings, static_cast<rune::DenseModel*>(rf.arch.get()), e2)) {
      std::printf("configure fail %s\n", e2.c_str());
      return 1;
    }
    float v;
    float w[3];
    dev.evaluateBoard(b, v, w);
    std::printf("%.9g %.9g %.9g %.9g\n", (double)v, (double)w[0], (double)w[1], (double)w[2]);
    return 0;
  }
  if (rf.isAdaptive || rf.isUncertainty) {
    rune::AdaptiveEvaluator aev;
    std::string e2;
    if (!aev.configure(&rf.varEmbeddings, static_cast<rune::AdaptiveModel*>(rf.arch.get()), e2)) {
      std::printf("configure fail %s\n", e2.c_str());
      return 1;
    }
    auto r = aev.evaluateBoard(b, rune::AdaptiveMode::Adaptive, thresholdOverride, useThreshold, false);
    std::printf("%.9g %.9g %.9g %.9g %.9g %d\n", (double)r.value, (double)r.wdl[0], (double)r.wdl[1],
                (double)r.wdl[2], (double)r.difficulty, r.refined ? 1 : 0);
    return 0;
  }
  rune::Evaluator ev(&rf.embeddings, rf.arch.get());
  auto r = ev.evaluateBoard(b);
  std::printf("%.9g %.9g %.9g %.9g\n", (double)r.value, (double)r.wdl[0], (double)r.wdl[1], (double)r.wdl[2]);
  return 0;
}
