#pragma once
#include <memory>
#include <string>
#include <vector>
#include "core/architectures/base/architecture.h"
#include "core/architectures/relational/relational.h"
namespace rune {
class RelLiteModel : public IArchitecture {
 public:
  RelLiteModel();
  void forward(const float* tokens, float& value, float* wdl, int phase = 1) const override;
  size_t parameterCount() const override;
  size_t modelSizeBytes() const override;
  const char* archId() const override { return "RUNE-REL-LITE"; }
  const char* archVersion() const override { return "0.1.0"; }
  void getTensors(std::vector<std::string>& names, std::vector<std::vector<int>>& shapes,
                  std::vector<const float*>& data) const override;
  bool setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) override;
  ModelSpec spec() const override;
  RelationalModel inner;
};
std::unique_ptr<RelLiteModel> createRelLite(std::string& err);
}
