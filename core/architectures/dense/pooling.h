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

#include <string>
#include <vector>

#include "core/architectures/dense/var_accum.h"

namespace rune {

enum class PoolMode { None, PerToken, Shared };

bool poolModeFromString(const std::string& name, PoolMode& out);
const char* poolModeName(PoolMode mode);

class TokenPool {
 public:
  TokenPool();

  bool configure(const VarWidths& widths, PoolMode mode, bool clipOut, int sharedWidth,
                 std::string& err);
  void forward(const float* in, float* out) const;
  size_t parameterCount() const;
  PoolMode mode() const { return mode_; }
  bool clipOut() const { return clip_; }

  std::vector<float> perW[8];
  std::vector<float> perB[8];
  std::vector<float> sharedS;
  std::vector<float> tokS[8];
  std::vector<float> tokB[8];

 private:
  VarWidths widths_;
  PoolMode mode_ = PoolMode::None;
  bool clip_ = true;
  int sharedWidth_ = 32;
  mutable std::vector<float> scratch_;
  static float clip01(float v);
};

class ChannelGate {
 public:
  ChannelGate();

  bool configure(const VarWidths& widths, bool enabled, std::string& err);
  void forward(const float* in, float* out) const;
  size_t parameterCount() const;
  bool enabled() const { return enabled_; }

  std::vector<float> ga;
  std::vector<float> gb;

 private:
  VarWidths widths_;
  bool enabled_ = false;
  std::vector<int> offsets_;
  static float clip01(float v);
};

}
