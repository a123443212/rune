#pragma once

#include <string>
#include <utility>
#include <vector>

#include "core/architectures/base/architecture.h"
#include "core/architectures/dense/pooling.h"
#include "core/architectures/dense/var_accum.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"
#include "core/simd/simd.h"

namespace rune {

enum class AdaptiveMode { Cheap, Always, Adaptive };

bool adaptiveModeFromString(const std::string& name, AdaptiveMode& out);
const char* adaptiveModeName(AdaptiveMode mode);

enum class SearchRoute { Difficulty, Uncertainty, Both, Full };

bool searchRouteFromString(const std::string& name, SearchRoute& out);
const char* searchRouteName(SearchRoute route);

struct RoutingThresholds {
  float diffT = 0.5f;
  float uncT = 0.5f;
  float stabT = 0.5f;
  bool hasTLow = false;
  float tLow = 0.5f;
};

struct AdaptiveBuildSpec {
  int dim = 32;
  std::string cheapPooling = "none";
  float alpha = 1.0f;
  float threshold = 0.5f;
  float tHigh = 0.5f;
  bool hasTLow = false;
  float tLow = 0.5f;
  std::vector<std::pair<int, int>> prunedPairs;
  std::string refinePrecision = "fp32";
  bool hasUncertainty = false;
  bool hasStabilityHead = false;
};

class AdaptiveModel : public IArchitecture {
 public:
  AdaptiveModel();

  bool configure(const AdaptiveBuildSpec& spec, std::string& err);
  void forward(const float* tokens, float& value, float* wdl) const override;
  void cheapForward(const float* acc, float* cheapFlat, float& value, float* wdl,
                    float& difficulty) const;
  void refineForward(const float* cheapFlat, float& value, float* wdl) const;
  float uncertaintyForward(const float* cheapFlat) const;
  float stabilityForward(const float* cheapFlat) const;
  bool route(float difficulty, bool prev) const;
  bool route(float difficulty, bool prev, float threshold) const;
  bool routeSearch(SearchRoute route, float difficulty, float uncertainty, float stability,
                   bool prev, const RoutingThresholds& t) const;
  size_t parameterCount() const override;
  size_t cheapParameterCount() const;
  size_t modelSizeBytes() const override { return parameterCount() * 4; }
  const char* archId() const override { return archId_.c_str(); }
  const char* archVersion() const override { return archVersion_.c_str(); }
  void getTensors(std::vector<std::string>& names, std::vector<std::vector<int>>& shapes,
                  std::vector<const float*>& data) const override;
  bool setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) override;
  ModelSpec spec() const override;
  const AdaptiveBuildSpec& buildSpec() const { return bspec_; }

  TokenPool pool;
  std::vector<float> cw1, cb1, cwv, cbv, cww, cbw;
  std::vector<float> dw, db;
  std::vector<float> uw, ub;
  std::vector<float> sw, sb;
  std::vector<float> wq, bq, wk, bk, wv, bv, gabS;
  std::vector<float> w1, b1, w2, b2, wvo, bvo, wwdl, bwdl;

 private:
  std::string archId_ = "RUNE-04";
  std::string archVersion_ = "0.4.0";
  AdaptiveBuildSpec bspec_;
  int dim_ = 32;
  int total_ = 256;
  bool pruneMask_[64];
  mutable std::vector<float> scratch_;
  static float clip01(float v);
};

struct AdaptiveEvalResult {
  float value = 0.0f;
  float wdl[3] = {0.0f, 0.0f, 0.0f};
  float difficulty = 0.0f;
  float uncertainty = 0.0f;
  float stability = 0.0f;
  bool refined = false;
};

class AdaptiveEvaluator {
 public:
  AdaptiveEvaluator();

  bool configure(VarEmbeddings* tables, AdaptiveModel* model, std::string& err);
  void refresh(const Board& board);
  void updateIncremental(const std::vector<ActiveFeature>& added,
                         const std::vector<ActiveFeature>& removed);
  AdaptiveEvalResult evaluate(AdaptiveMode mode, float thresholdOverride, bool useOverride,
                              bool prev) const;
  AdaptiveEvalResult evaluateBoard(const Board& board, AdaptiveMode mode,
                                                    float thresholdOverride, bool useOverride,
                                                    bool prev);
  AdaptiveEvalResult evaluateSearch(const Board& board, SearchRoute route,
                                    const RoutingThresholds& t, bool prev);
  void currentTokens(float* out) const;
  void currentCheap(float* out) const;

 private:
  VarAccumulator acc_;
  VarEmbeddings* tables_ = nullptr;
  AdaptiveModel* model_ = nullptr;
  mutable std::vector<float> tokenBuf_;
  mutable std::vector<float> cheapBuf_;
};

}
