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

#include <cstdio>
#include <string>
#include "core/compiler/artifact.h"
int main(int argc, char** argv) {
  std::string model;
  for (int i = 1; i + 1 < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model") model = argv[i + 1];
  }
  if (model.empty()) {
    std::printf("usage: rune_compile_support --model <path>\n");
    return 1;
  }
  std::string header;
  rune::vart::CompiledInfo info;
  std::string err;
  bool ok = rune::vart::readCompiledHeader(model, header, info, err);
  if (!ok) {
    std::printf("artifact INVALID %s err %s\n", model.c_str(), err.c_str());
    return 2;
  }
  std::printf("artifact %s\n", model.c_str());
  std::printf("compiled %d\n", (int)info.compiled);
  std::printf("ir %s\n", info.irVersion.c_str());
  std::printf("compiler %s\n", info.compilerVersion.c_str());
  std::printf("isa %s\n", info.targetIsa.c_str());
  std::printf("cpu %s\n", info.targetCpu.c_str());
  std::printf("source_hash %s\n", info.sourceHash.c_str());
  std::printf("plan_hash %s\n", info.planHash.c_str());
  std::printf("model_hash %s\n", info.modelHash.c_str());
  std::printf("kernels %d\n", info.kernelCount);
  std::printf("arena_bytes %zu\n", info.arenaBytes);
  return 0;
}
