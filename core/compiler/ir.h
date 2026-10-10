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
namespace rune {
namespace vir {
struct IrModel {
  std::string architecture;
  std::string archVersion;
  int tokens = 8;
  int tokenDim = 32;
  std::string dtype = "fp32";
  std::string quantization = "fp32";
  std::string gate = "clip";
  float alpha = 1.0f;
  int headH1 = 128;
  int headH2 = 32;
  float threshold = 0.5f;
  float tHigh = 0.5f;
  bool hasTLow = false;
  float tLow = 0.5f;
  bool adaptive = false;
};
struct IrTensor {
  std::string name;
  std::vector<int> shape;
  std::string dtype;
  std::string layout;
  bool constant = true;
  int birth = 0;
  int death = 12;
};
struct IrOp {
  std::string id;
  std::string kind;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};
struct IrTarget {
  std::string cpu = "generic-x86-64";
  std::string isa = "portable";
  int vectorWidth = 1;
  std::string dtype = "fp32";
  std::string quantization = "fp32";
};
struct KernelEntry {
  std::string op;
  std::string kind;
  std::string kernelId;
  std::string shape;
  std::string dtype;
  std::string packing;
  std::string isa;
  std::string fusionGroup;
};
struct FusionEntry {
  std::string id;
  std::vector<std::string> ops;
  std::string kernel;
};
struct RuneIr {
  std::string irVersion = "1.0";
  IrModel model;
  std::vector<IrTensor> tensors;
  std::vector<IrOp> ops;
  IrTarget target;
  std::vector<KernelEntry> kernels;
  std::vector<FusionEntry> fusion;
  size_t arenaBytes = 0;
};
bool verifyIr(const RuneIr& ir, std::string& err);
std::string shapeKey(int tokens, int dim, int h1, int h2, const std::string& kind);
std::string kernelFor(const std::string& kind, const std::string& shape);
}
}
