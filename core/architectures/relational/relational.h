#pragma once

#include <string>
#include <vector>

#include "core/accumulators/flex_accumulator.h"
#include "core/architectures/base/architecture.h"
#include "core/architectures/relational/context.h"
#include "core/board/board.h"
#include "core/simd/simd.h"

namespace rune {

struct RelationalConfig {
  int tokens = 8;
  int dim = 32;
  GateFn gate = GateFn::Clip;
  float alpha = 1.0f;
  bool dynamicBias = false;
  static constexpr int kHeadH1 = 128;
  static constexpr int kHeadH2 = 32;
};

class RelationalMixer {
 public:
  RelationalMixer();

  bool configure(const RelationalConfig& cfg, std::string& err);
  void forward(const float* x, const float* ctx, float* out) const;
  size_t parameterCount() const;
  const RelationalConfig& config() const { return cfg_; }

  std::vector<float> wq, bq, wk, bk, wv, bv, gabS, dynU, dynW;

 private:
  RelationalConfig cfg_;
  mutable std::vector<float> scratch_;
};

class RelationalModel : public IArchitecture {
 public:
  RelationalModel();

  bool configure(const RelationalConfig& cfg, std::string& err);
  void forward(const float* tokens, float& value, float* wdl, int phase = 1) const override;
  void forwardWithContext(const float* tokens, const float* ctx, float& value, float* wdl, int phase = 1) const;
  size_t parameterCount() const override;
  size_t modelSizeBytes() const override { return parameterCount() * 4; }
  const char* archId() const override { return "RUNE-REL-02"; }
  const char* archVersion() const override { return "0.2.0"; }
  void getTensors(std::vector<std::string>& names, std::vector<std::vector<int>>& shapes,
                  std::vector<const float*>& data) const override;
  bool setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) override;
  ModelSpec spec() const override;
  const RelationalConfig& config() const { return mixer.config(); }

  RelationalMixer mixer;
  std::vector<HeadBucket> heads_;

 private:
  const HeadBucket& headFor(int phase) const;
  mutable std::vector<float> scratch_;
};

struct EvalResultFlex {
  float value = 0.0f;
  float wdl[3] = {0.0f, 0.0f, 0.0f};
};

class RelationalEvaluator {
 public:
  RelationalEvaluator();

  bool configure(FlexEmbeddings* tables, TokenLayout* layout, RelationalModel* model,
                 std::string& err);
  void refresh(const Board& board);
  void updateIncremental(const Board& board, const std::vector<ActiveFeature>& added,
                         const std::vector<ActiveFeature>& removed);
  EvalResultFlex evaluate() const;
  EvalResultFlex evaluateBoard(const Board& board);
  void currentTokens(float* out) const;
  void currentContext(float* out) const;

 private:
  FlexAccumulator acc_;
  FlexEmbeddings* tables_ = nullptr;
  TokenLayout* layout_ = nullptr;
  RelationalModel* model_ = nullptr;
  float ctx_[ContextSpec::kDim];
  int phase_ = 1;
  mutable std::vector<float> tokenBuf_;
};

}
