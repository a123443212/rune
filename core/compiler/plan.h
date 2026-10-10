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
#include "core/compiler/ir.h"
namespace rune {
namespace vplan {
void selectKernels(vir::RuneIr& ir);
size_t planArenaBytes(int tokens, int dim, int h1, int h2);
double kernelCost(const std::string& shape, const std::string& kernelId, const std::string& isa);
std::string fusionOf(const std::string& kind);
}
}
