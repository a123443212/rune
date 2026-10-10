#pragma once
#include <vector>
#include "core/architectures/attention/attention.h"
#include "core/architectures/base/architecture.h"
namespace rune {
class RuneDualAttentionModel : public IArchitecture {
 public:
  RuneDualAttentionModel();
  explicit RuneDualAttentionModel(GateFn gate);
  void forward(const float* tokens, float& value, float* wdl, int phase) const override;
  size_t parameterCount() const override;
  size_t modelSizeBytes() const override;
  const char* archId() const override { return "RUNE-ATTN-DUAL"; }
  const char* archVersion() const override { return "0.1.0"; }
  void getTensors(std::vector<std::string>& names, std::vector<std::vector<int>>& shapes,
                  std::vector<const float*>& data) const override;
  bool setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) override;
  ModelSpec spec() const override;
  RuneAttentionBlock attn1;
  RuneAttentionBlock attn2;
  std::vector<HeadBucket> heads_;
 private:
  const HeadBucket& headFor(int phase) const;
  GateFn gate_;
  mutable std::vector<float> scratch_;
};
}
