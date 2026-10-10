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

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "core/architectures/base/architecture.h"
#include "core/simd/simd.h"

namespace rune {

inline float swigluSilu(float x) { return x / (1.0f + std::exp(-x)); }

inline bool headIsSwiGlu(const HeadBucket& h) { return !h.wgate.empty(); }

inline void swigluForward(const HeadBucket& h, const float* flat, int flatN, float& value, float* wdl,
                          std::vector<float>& scratch) {
  int h1n = static_cast<int>(h.bgate.size());
  int h2n = static_cast<int>(h.b2.size());
  size_t need = static_cast<size_t>(h1n) * 3 + static_cast<size_t>(h2n);
  if (scratch.size() < need) scratch.resize(need);
  float* gate = scratch.data();
  float* up = scratch.data() + h1n;
  float* act = scratch.data() + h1n * 2;
  float* h2 = scratch.data() + h1n * 3;
  simd::matVec(h.wgate.data(), flat, h.bgate.data(), gate, h1n, flatN);
  simd::matVec(h.wup.data(), flat, h.bup.data(), up, h1n, flatN);
  for (int i = 0; i < h1n; ++i) act[i] = swigluSilu(gate[i]) * up[i];
  simd::matVecClipped(h.w2.data(), act, h.b2.data(), h2, h2n, h1n);
  float v = h.bvo[0];
  for (int i = 0; i < h2n; ++i) v += h.wvo[i] * h2[i];
  value = std::tanh(v);
  simd::matVec(h.wwdl.data(), h2, h.bwdl.data(), wdl, 3, h2n);
}

inline void headWantNames(std::vector<std::string>& want, size_t nbuckets, bool swiglu,
                          const char* vkey, const char* bkey) {
  for (size_t b = 0; b < nbuckets; ++b) {
    std::string suffix = nbuckets > 1 ? "_b" + std::to_string(b) : "";
    if (swiglu) {
      const char* keys[10] = {"wgate", "bgate", "wup", "bup", "w2", "b2", vkey, bkey, "wwdl", "bwdl"};
      for (size_t k = 0; k < 10; ++k) want.push_back(std::string(keys[k]) + suffix);
    } else {
      const char* keys[8] = {"w1", "b1", "w2", "b2", vkey, bkey, "wwdl", "bwdl"};
      for (size_t k = 0; k < 8; ++k) want.push_back(std::string(keys[k]) + suffix);
    }
  }
}

inline bool shapeIsMat(const std::vector<int>& s, int r, int c) {
  return s.size() == 2 && s[0] == r && s[1] == c;
}

inline bool shapeIsVec(const std::vector<int>& s, int n) {
  return s.size() == 1 && s[0] == n;
}

inline bool copySlot(std::vector<float>& dst, size_t n, const std::vector<float>& flat, size_t& off) {
  dst.assign(n, 0.0f);
  if (off + n > flat.size()) return false;
  for (size_t i = 0; i < n; ++i) dst[i] = flat[off + i];
  off += n;
  return true;
}

inline bool readHeadBucket(const std::vector<std::vector<int>>& shapes, const std::vector<float>& flat,
                           size_t o, bool swiglu, int inDim, HeadBucket& h, size_t& off,
                           int& h1out, int& h2out) {
  if (o >= shapes.size()) return false;
  if (!swiglu) {
    if (o + 8 > shapes.size()) return false;
    if (shapes[o].size() != 2 || shapes[o + 3].size() != 1) return false;
    int h1 = shapes[o][0];
    int h2 = shapes[o + 3][0];
    if (h1 < 1 || h1 > 4096 || h2 < 1 || h2 > 4096) return false;
    if (!shapeIsMat(shapes[o], h1, inDim)) return false;
    if (!shapeIsVec(shapes[o + 1], h1)) return false;
    if (shapes[o + 2].size() != 2 || shapes[o + 2][0] != h2) return false;
    int w2c = shapes[o + 2][1];
    bool pair = (w2c == 2 * h1);
    if (w2c != h1 && !pair) return false;
    if (!shapeIsVec(shapes[o + 3], h2)) return false;
    if (!shapeIsMat(shapes[o + 4], 1, h2)) return false;
    if (!shapeIsVec(shapes[o + 5], 1)) return false;
    if (!shapeIsMat(shapes[o + 6], 3, h2)) return false;
    if (!shapeIsVec(shapes[o + 7], 3)) return false;
    size_t w2n = static_cast<size_t>(h2) * static_cast<size_t>(pair ? 2 * h1 : h1);
    if (!copySlot(h.w1, static_cast<size_t>(h1) * static_cast<size_t>(inDim), flat, off)) return false;
    if (!copySlot(h.b1, static_cast<size_t>(h1), flat, off)) return false;
    if (!copySlot(h.w2, w2n, flat, off)) return false;
    if (!copySlot(h.b2, static_cast<size_t>(h2), flat, off)) return false;
    if (!copySlot(h.wvo, static_cast<size_t>(h2), flat, off)) return false;
    if (!copySlot(h.bvo, 1, flat, off)) return false;
    if (!copySlot(h.wwdl, static_cast<size_t>(3 * h2), flat, off)) return false;
    if (!copySlot(h.bwdl, 3, flat, off)) return false;
    h1out = h1;
    h2out = h2;
    return true;
  }
  if (o + 10 > shapes.size()) return false;
  if (shapes[o].size() != 2 || shapes[o + 1].size() != 1) return false;
  int h1 = shapes[o][0];
  int h2 = shapes[o + 5].size() == 1 ? shapes[o + 5][0] : -1;
  if (h1 < 1 || h1 > 4096 || h2 < 1 || h2 > 4096) return false;
  if (!shapeIsMat(shapes[o], h1, inDim)) return false;
  if (!shapeIsVec(shapes[o + 1], h1)) return false;
  if (!shapeIsMat(shapes[o + 2], h1, inDim)) return false;
  if (!shapeIsVec(shapes[o + 3], h1)) return false;
  if (!shapeIsMat(shapes[o + 4], h2, h1)) return false;
  if (!shapeIsVec(shapes[o + 5], h2)) return false;
  if (!shapeIsMat(shapes[o + 6], 1, h2)) return false;
  if (!shapeIsVec(shapes[o + 7], 1)) return false;
  if (!shapeIsMat(shapes[o + 8], 3, h2)) return false;
  if (!shapeIsVec(shapes[o + 9], 3)) return false;
  if (!copySlot(h.wgate, static_cast<size_t>(h1) * static_cast<size_t>(inDim), flat, off)) return false;
  if (!copySlot(h.bgate, static_cast<size_t>(h1), flat, off)) return false;
  if (!copySlot(h.wup, static_cast<size_t>(h1) * static_cast<size_t>(inDim), flat, off)) return false;
  if (!copySlot(h.bup, static_cast<size_t>(h1), flat, off)) return false;
  if (!copySlot(h.w2, static_cast<size_t>(h2) * static_cast<size_t>(h1), flat, off)) return false;
  if (!copySlot(h.b2, static_cast<size_t>(h2), flat, off)) return false;
  if (!copySlot(h.wvo, static_cast<size_t>(h2), flat, off)) return false;
  if (!copySlot(h.bvo, 1, flat, off)) return false;
  if (!copySlot(h.wwdl, static_cast<size_t>(3 * h2), flat, off)) return false;
  if (!copySlot(h.bwdl, 3, flat, off)) return false;
  h1out = h1;
  h2out = h2;
  return true;
}

}
