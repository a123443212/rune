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

#include "core/accumulators/token_layout.h"

#include "core/features/feature_set.h"

namespace rune {

namespace {

void addRange(std::vector<TokenSource>& v, int group) {
  TokenSource s;
  s.group = static_cast<uint8_t>(group);
  s.lo = 0;
  s.hi = static_cast<uint16_t>(GroupedFeatureSet::vocabSize(group));
  v.push_back(s);
}

void addSplit(std::vector<TokenSource>& v, int group, int lo, int hi) {
  TokenSource s;
  s.group = static_cast<uint8_t>(group);
  s.lo = static_cast<uint16_t>(lo);
  s.hi = static_cast<uint16_t>(hi);
  v.push_back(s);
}

}  // namespace

bool TokenLayout::make(int tokens, int dim, TokenLayout& out, std::string& err) {
  if (dim != 24 && dim != 32 && dim != 40) {
    err = "unsupported token dim";
    return false;
  }
  out.tokens = tokens;
  out.dim = dim;
  out.sources.clear();
  if (tokens == 8) {
    for (int g = 0; g < 8; ++g) {
      std::vector<TokenSource> v;
      addRange(v, g);
      if (g == 0) addRange(v, 8);
      out.sources.push_back(v);
    }
    return true;
  }
  if (tokens == 6) {
    for (int g = 0; g <= 2; ++g) {
      std::vector<TokenSource> v;
      addRange(v, g);
      if (g == 0) addRange(v, 8);
      out.sources.push_back(v);
    }
    std::vector<TokenSource> majors;
    addRange(majors, 3);
    addRange(majors, 4);
    out.sources.push_back(majors);
    std::vector<TokenSource> tactical;
    addRange(tactical, 5);
    addRange(tactical, 6);
    out.sources.push_back(tactical);
    std::vector<TokenSource> global;
    addRange(global, 7);
    out.sources.push_back(global);
    return true;
  }
  if (tokens == 10) {
    for (int g = 0; g <= 1; ++g) {
      std::vector<TokenSource> v;
      addRange(v, g);
      if (g == 0) addRange(v, 8);
      out.sources.push_back(v);
    }
    std::vector<TokenSource> knights;
    addSplit(knights, 2, 0, 128);
    out.sources.push_back(knights);
    std::vector<TokenSource> bishops;
    addSplit(bishops, 2, 128, 256);
    out.sources.push_back(bishops);
    for (int g = 3; g <= 4; ++g) {
      std::vector<TokenSource> v;
      addRange(v, g);
      out.sources.push_back(v);
    }
    std::vector<TokenSource> victims;
    addSplit(victims, 5, 0, 384);
    out.sources.push_back(victims);
    std::vector<TokenSource> attackers;
    addSplit(attackers, 5, 384, 512);
    out.sources.push_back(attackers);
    for (int g = 6; g <= 7; ++g) {
      std::vector<TokenSource> v;
      addRange(v, g);
      out.sources.push_back(v);
    }
    return true;
  }
  err = "unsupported token count";
  return false;
}

int TokenLayout::findToken(uint8_t group, uint16_t index) const {
  for (size_t t = 0; t < sources.size(); ++t) {
    for (const TokenSource& s : sources[t]) {
      if (s.group == group && index >= s.lo && index < s.hi) return static_cast<int>(t);
    }
  }
  return -1;
}

}
