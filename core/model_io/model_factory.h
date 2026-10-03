#pragma once

#include <memory>
#include <string>

#include "core/architectures/adaptive/adaptive.h"
#include "core/architectures/base/architecture.h"
#include "core/architectures/dense/dense.h"
#include "core/architectures/relational/relational.h"

namespace rune {

std::unique_ptr<IArchitecture> createArchitecture(const std::string& archId);

struct FlexBuildSpec {
  int tokens = 8;
  int dim = 32;
  std::string gate = "clip";
  float alpha = 1.0f;
  bool dynamicBias = false;
};

std::unique_ptr<RelationalModel> createRelational(const FlexBuildSpec& spec, std::string& err);

bool isSupportedVersion(const std::string& version);

std::unique_ptr<DenseModel> createDense(const DenseBuildSpec& spec, std::string& err);

std::unique_ptr<AdaptiveModel> createAdaptive(const AdaptiveBuildSpec& spec, std::string& err);

}
