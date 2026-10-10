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
#include <cstdint>
#include <string>
#include <vector>
#include "core/features/feature_set.h"
namespace rune {
struct IntermediateDump {
  std::vector<ActiveFeature> features;
  std::vector<float> accumulator;
  std::vector<float> tokens;
  std::vector<float> q;
  std::vector<float> k;
  std::vector<float> v;
  std::vector<float> scores;
  std::vector<float> gates;
  std::vector<float> mixed;
  std::vector<float> h1;
  std::vector<float> h2;
  float value = 0.0f;
  float wdl[3] = {0.0f, 0.0f, 0.0f};
  bool refine = false;
  float difficulty = 0.0f;
  bool writeText(const std::string& path) const;
};
}
