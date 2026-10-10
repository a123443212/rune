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
#include "core/features/feature_set.h"
#include "core/accumulators/grouped_accumulator.h"
namespace rune {
namespace aspec {
struct GroupOffsets {
  int dim = 32;
  int off[9] = {0, 32, 64, 96, 128, 160, 192, 224, 0};
};
GroupOffsets offsetsFor(int dim);
void refreshGrouped(float* acc, const EmbeddingTables* t, const std::vector<ActiveFeature>& feats, const GroupOffsets& g);
void applyGrouped(float* acc, const EmbeddingTables* t, const std::vector<ActiveFeature>& added, const std::vector<ActiveFeature>& removed, const GroupOffsets& g);
int packIds(const std::vector<ActiveFeature>& feats, uint8_t* groups, uint16_t* idx, int cap);
}
}
