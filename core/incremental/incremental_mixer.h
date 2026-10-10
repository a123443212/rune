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

#include "core/incremental/relational_cache.h"

namespace rune {
namespace v13 {

void denseBaselineForward(const IncrWeights& w, const float* tokens, const float* ctx,
                          float* out);

void qkvRowsForward(const IncrWeights& w, const float* tokens,
                    const std::vector<int>& changed, float* q, float* k, float* vv);

int scoreCellsForChanged(int tokens, const std::vector<int>& changed);

}
}
