#pragma once

#include <string>
#include <vector>

#include "core/architectures/base/architecture.h"
#include "core/architectures/dense/pooling.h"
#include "core/architectures/dense/var_accum.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"
#include "core/simd/simd.h"

namespace rune {

struct DenseBuildSpec {
  std::string variant = "A";
  std::vector<int> dims = {32, 32, 32, 32, 32, 32, 32, 32};
  std::string pooling = "none";
  bool poolClip = true;
  bool gateOn = false;
  int sharedWidth = 32;
  int headH1 = 128;
  int headH2 = 32;
};

class DenseModel : public IArchitecture {
 public:
  DenseModel();

  bool configure(const DenseBuildSpec& spec, std::string& err);
  void forward(const float* tokens, float& value, float* wdl, int phase = 1) const override;
  size_t parameterCount() const override;
  size_t modelSizeBytes() const override { return parameterCount() * 4; }
  const char* archId() const override { return archId_.c_str(); }
  const char* archVersion() const override { return "0.3.0"; }
  void getTensors(std::vector<std::string>& names, std::vector<std::vector<int>>& shapes,
                  std::vector<const float*>& data) const override;
  bool setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) override;
  ModelSpec spec() const override;
  const DenseBuildSpec& buildSpec() const { return bspec_; }

  TokenPool pool;
  ChannelGate gate;
  std::vector<float> w1, b1, w2, b2, wvo, bvo, wwdl, bwdl;

 private:
  std::string archId_ = "RUNE-03-A";
  DenseBuildSpec bspec_;
  VarWidths widths_;
  int headH1_ = 128;
  int headH2_ = 32;
  mutable std::vector<float> scratch_;
};

class DenseEvaluator {
 public:
  DenseEvaluator();

  bool configure(VarEmbeddings* tables, DenseModel* model, std::string& err);
  void refresh(const Board& board);
  void updateIncremental(const std::vector<ActiveFeature>& added,
                         const std::vector<ActiveFeature>& removed);
  void evaluate(float& value, float* wdl) const;
  void evaluateBoard(const Board& board, float& value, float* wdl);
  void currentTokens(float* out) const;

 private:
  VarAccumulator acc_;
  VarEmbeddings* tables_ = nullptr;
  DenseModel* model_ = nullptr;
  mutable std::vector<float> tokenBuf_;
};

}
