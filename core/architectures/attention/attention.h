#pragma once

#include <vector>

#include "core/architectures/base/architecture.h"
#include "core/simd/simd.h"

namespace rune {

class RuneAttentionBlock {
 public:
  static constexpr int kTokens = 8;
  static constexpr int kDim = 32;

  explicit RuneAttentionBlock(bool useGab);

  void forward(const float* x, float* out) const;
  size_t parameterCount() const;
  bool usesGab() const { return useGab_; }

  std::vector<float> wq, bq, wk, bk, wv, bv, gab;

 private:
  bool useGab_;
  mutable std::vector<float> scratch_;
};

class RuneAttnModel : public IArchitecture {
 public:
  RuneAttnModel();
  explicit RuneAttnModel(bool useGab);

  void forward(const float* tokens, float& value, float* wdl, int phase) const override;
  size_t parameterCount() const override;
  size_t modelSizeBytes() const override;
  const char* archId() const override { return useGab_ ? "RUNE-ATTN-GAB" : "RUNE-ATTN"; }
  const char* archVersion() const override { return "0.1.0"; }
  void getTensors(std::vector<std::string>& names, std::vector<std::vector<int>>& shapes,
                  std::vector<const float*>& data) const override;
  bool setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) override;
  ModelSpec spec() const override;

  RuneAttentionBlock attn;
  std::vector<HeadBucket> heads_;

 private:
  const HeadBucket& headFor(int phase) const;
  bool useGab_;
  mutable std::vector<float> scratch_;
};

}
