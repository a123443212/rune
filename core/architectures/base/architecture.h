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
#include <utility>
#include <vector>

namespace rune {

struct ModelSpec {
  std::string arch;
  std::string archVersion = "0.1.0";
  std::string game = "chess";
  std::string featureSet = "grouped_hkav2_fullthreats_v02";
  int tokens = 8;
  int tokenDim = 32;
  std::string attention = "none";
  std::string geometricBias = "none";
  std::string head = "value_wdl";
  std::string quantization = "fp32";
  std::string gate = "clip";
  float alpha = 1.0f;
  std::string variant;
  std::vector<int> tokenDims;
  std::string pooling = "none";
  bool gateOn = false;
  bool poolClip = true;
  std::string cheapPooling = "none";
  float threshold = 0.5f;
  float tHigh = 0.5f;
  bool hasTLow = false;
  float tLow = 0.5f;
  std::string refinePrecision = "fp32";
  std::vector<std::pair<int, int>> prunedPairs;
  bool hasUncertainty = false;
  bool hasStabilityHead = false;
  int headH1 = 128;
  int headH2 = 32;
  int cheapHidden = 32;
  int refH1 = 128;
  int refH2 = 32;
  int sharedWidth = 32;
  int headBuckets = 1;

  std::string canonicalString() const;
  uint64_t configHash() const;
};

class IArchitecture {
 public:
  virtual ~IArchitecture() = default;
  virtual void forward(const float* tokens, float& value, float* wdl, int phase = 1) const = 0;
  virtual size_t parameterCount() const = 0;
  virtual size_t modelSizeBytes() const = 0;
  virtual const char* archId() const = 0;
  virtual const char* archVersion() const = 0;
  virtual void getTensors(std::vector<std::string>& names, std::vector<std::vector<int>>& shapes,
                          std::vector<const float*>& data) const = 0;
  virtual bool setTensors(const std::vector<std::string>& names, const std::vector<float>& flat) = 0;
  virtual ModelSpec spec() const = 0;
};

struct HeadBucket {
  std::vector<float> w1, b1, w2, b2, wvo, bvo, wwdl, bwdl;
};

uint64_t fnv1aHash(const std::string& s);
uint64_t fnv1aHash(const uint8_t* data, size_t n);

enum class GateFn { Clip, HardSigmoid, Screlu };

bool gateFromString(const std::string& name, GateFn& out);
const char* gateName(GateFn fn);
float applyGate(GateFn fn, float s);

}
