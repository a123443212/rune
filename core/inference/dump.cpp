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

#include "core/inference/dump.h"
#include <cstdio>
namespace rune {
bool IntermediateDump::writeText(const std::string& path) const {
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) return false;
  std::fprintf(f, "features %zu\n", features.size());
  for (size_t i = 0; i < features.size(); ++i)
    std::fprintf(f, "f %u %u\n", (unsigned)features[i].group, (unsigned)features[i].index);
  std::fprintf(f, "accumulator %zu\n", accumulator.size());
  for (size_t i = 0; i < accumulator.size(); ++i) std::fprintf(f, "a %.9g\n", (double)accumulator[i]);
  std::fprintf(f, "tokens %zu\n", tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) std::fprintf(f, "t %.9g\n", (double)tokens[i]);
  std::fprintf(f, "q %zu\n", q.size());
  for (float x : q) std::fprintf(f, "q %.9g\n", (double)x);
  std::fprintf(f, "k %zu\n", k.size());
  for (float x : k) std::fprintf(f, "k %.9g\n", (double)x);
  std::fprintf(f, "v %zu\n", v.size());
  for (float x : v) std::fprintf(f, "v %.9g\n", (double)x);
  std::fprintf(f, "scores %zu\n", scores.size());
  for (float x : scores) std::fprintf(f, "s %.9g\n", (double)x);
  std::fprintf(f, "gates %zu\n", gates.size());
  for (float x : gates) std::fprintf(f, "g %.9g\n", (double)x);
  std::fprintf(f, "mixed %zu\n", mixed.size());
  for (float x : mixed) std::fprintf(f, "m %.9g\n", (double)x);
  std::fprintf(f, "h1 %zu\n", h1.size());
  for (float x : h1) std::fprintf(f, "h1 %.9g\n", (double)x);
  std::fprintf(f, "h2 %zu\n", h2.size());
  for (float x : h2) std::fprintf(f, "h2 %.9g\n", (double)x);
  std::fprintf(f, "value %.9g\n", (double)value);
  std::fprintf(f, "wdl %.9g %.9g %.9g\n", (double)wdl[0], (double)wdl[1], (double)wdl[2]);
  std::fprintf(f, "refine %d difficulty %.9g\n", refine ? 1 : 0, (double)difficulty);
  std::fclose(f);
  return true;
}
}
