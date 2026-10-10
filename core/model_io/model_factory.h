/*
RUNE — Relational Unified Neural Evaluator
Copyright (C) 2026 a123443212

SPDX-License-Identifier: MIT OR Apache-2.0

This project is dual-licensed under the MIT License and the
Apache License, Version 2.0. You may choose either license
when using, copying, modifying, or distributing this software.

MIT License: https://opensource.org/license/mit
Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, this
software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
*/

#pragma once

#include <memory>
#include <string>

#include "core/architectures/adaptive/adaptive.h"
#include "core/architectures/base/architecture.h"
#include "core/architectures/dense/dense.h"
#include "core/architectures/relational/relational.h"

namespace rune {

std::unique_ptr<IArchitecture> createArchitecture(const std::string& archId,
                                                   GateFn gate = GateFn::Clip);

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
