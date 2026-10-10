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
namespace rune {
namespace vart {
struct CompiledInfo {
  bool compiled = false;
  std::string irVersion;
  std::string compilerVersion;
  std::string targetIsa;
  std::string targetCpu;
  std::string sourceHash;
  std::string planHash;
  std::string modelHash;
  size_t arenaBytes = 0;
  int kernelCount = 0;
};
bool readCompiledHeader(const std::string& path, std::string& headerOut, CompiledInfo& info, std::string& err);
bool checkIsaSupported(const std::string& isa, std::string& err);
uint64_t fnv1a64Bytes(const uint8_t* data, size_t n);
std::string cacheKeyFor(const std::string& src, const std::string& arch, const std::string& quant, const std::string& isa);
}
}
