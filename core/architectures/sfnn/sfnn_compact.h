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
#include <vector>
#include "core/architectures/base/architecture.h"
namespace rune {
class SfnnCompact : public IArchitecture {
 public:
  static constexpr int kIn = 256;
  static constexpr int kH1 = 128;
  static constexpr int kH2 = 16;
  SfnnCompact();
  void forward(const float* tokens, float& value, float* wdl, int phase = 1) const override;
  size_t parameterCount() const override;
  size_t modelSizeBytes() const override;
  const char* archId() const override { return "RUNE-SFNN-C"; }
  const char* archVersion() const override { return "0.1.0"; }
  void getTensors(std::vector<std::string>& names, std::vector<std::vector<int>>& shapes,
                  std::vector<const float*>& data) const override;
  bool setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) override;
  ModelSpec spec() const override;
  std::vector<HeadBucket> heads_;
 private:
  const HeadBucket& headFor(int phase) const;
  mutable std::vector<float> scratch_;
};
}
