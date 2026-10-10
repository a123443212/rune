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
namespace rune {
namespace eng {
struct EvalOutput {
  float value = 0.0f;
  float wdl[3] = {0.0f, 1.0f, 0.0f};
  float uncertainty = 0.0f;
  bool refine = false;
};
float canonicalValue(float raw);
int engineScore(float value);
void normalizeWdl(float w[3]);
float wdlValue(const float w[3]);
bool valueWdlConsistent(float value, const float w[3], float tol);
bool isExtreme(float value, float bound);
std::string evalCacheKey(const std::string& fen, const std::string& modelHash, const std::string& mode);
}
}
