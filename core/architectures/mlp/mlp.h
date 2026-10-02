#pragma once

#include <vector>

#include "core/architectures/base/architecture.h"
#include "core/simd/simd.h"

namespace rune {

class GroupedMlp : public IArchitecture {
 public:
  static constexpr int kIn = 256;
  static constexpr int kH1 = 128;
  static constexpr int kH2 = 32;

  GroupedMlp();

  void forward(const float* tokens, float& value, float* wdl) const override;
  size_t parameterCount() const override;
  size_t modelSizeBytes() const override;
  const char* archId() const override { return "RUNE-MLP"; }
  const char* archVersion() const override { return "0.1.0"; }
  void getTensors(std::vector<std::string>& names, std::vector<std::vector<int>>& shapes,
                  std::vector<const float*>& data) const override;
  bool setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) override;
  ModelSpec spec() const override;

  std::vector<float> w1, b1, w2, b2, wv, bv, wwdl, bwdl;

 private:
  mutable std::vector<float> scratch_;
};

class SfnnBaseline : public IArchitecture {
 public:
  static constexpr int kIn = 256;
  static constexpr int kH1 = 256;
  static constexpr int kH2 = 32;

  SfnnBaseline();

  void forward(const float* tokens, float& value, float* wdl) const override;
  size_t parameterCount() const override;
  size_t modelSizeBytes() const override;
  const char* archId() const override { return "RUNE-SFNN"; }
  const char* archVersion() const override { return "0.1.0"; }
  void getTensors(std::vector<std::string>& names, std::vector<std::vector<int>>& shapes,
                  std::vector<const float*>& data) const override;
  bool setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) override;
  ModelSpec spec() const override;

  std::vector<float> w1, b1, w2, b2, wv, bv, wwdl, bwdl;

 private:
  mutable std::vector<float> scratch_;
};

}
