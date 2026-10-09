#pragma once

#include <string>
#include <vector>

#include "core/architectures/base/architecture.h"
#include "core/simd/simd.h"

namespace rune {

class MultiHeadMixer {
 public:
  static constexpr int kTokens = 8;
  static constexpr int kDim = 32;
  static constexpr int kHeads = 4;
  static constexpr int kHeadDim = 8;

  MultiHeadMixer(GateFn gate = GateFn::Clip);

  void forward(const float* x, float* out) const;
  size_t parameterCount() const;

  std::vector<float> wq[kHeads];
  std::vector<float> bq[kHeads];
  std::vector<float> wk[kHeads];
  std::vector<float> bk[kHeads];
  std::vector<float> wv[kHeads];
  std::vector<float> bv[kHeads];
  std::vector<float> gab[kHeads];
  std::vector<float> wo;
  std::vector<float> bwo;

 private:
  GateFn gate_;
  mutable std::vector<float> scratch_;
};

class RuneAttnMhModel : public IArchitecture {
 public:
  RuneAttnMhModel(GateFn gate = GateFn::Clip);

  void forward(const float* tokens, float& value, float* wdl, int phase) const override;
  size_t parameterCount() const override;
  size_t modelSizeBytes() const override { return parameterCount() * 4; }
  const char* archId() const override { return "RUNE-ATTN-MH4"; }
  const char* archVersion() const override { return "0.1.0"; }
  void getTensors(std::vector<std::string>& names, std::vector<std::vector<int>>& shapes,
                  std::vector<const float*>& data) const override;
  bool setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) override;
  ModelSpec spec() const override;

  MultiHeadMixer mixer;
  std::vector<HeadBucket> heads_;

 private:
  const HeadBucket& headFor(int phase) const;
  GateFn gate_;
  mutable std::vector<float> scratch_;
};

}
