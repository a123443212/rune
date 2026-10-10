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
#include <vector>

#include "core/accumulators/grouped_accumulator.h"
#include "core/accumulators/token_layout.h"
#include "core/features/feature_set.h"

namespace rune {
namespace v13 {

struct GroupDelta {
  std::vector<ActiveFeature> added[GroupedFeatureSet::kNumGroups];
  std::vector<ActiveFeature> removed[GroupedFeatureSet::kNumGroups];
  std::vector<int> changedGroups;
};

GroupDelta detectChangedGroups(const std::vector<ActiveFeature>& before,
                               const std::vector<ActiveFeature>& after);

std::vector<int> changedTokensFromGroups(const GroupDelta& delta,
                                         const int* tokenOfGroup, int numTokens);

std::vector<int> changedTokensByCompare(const float* tokOld, const float* tokNew,
                                        int numTokens, int dim);

void applyAccumDelta(const EmbeddingTables& tables, float* acc, const TokenLayout& layout,
                     const GroupDelta& delta);

}
}
