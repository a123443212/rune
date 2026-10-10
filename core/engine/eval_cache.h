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
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>
namespace rune {
namespace eng {
class EvalCache {
 public:
  EvalCache(const std::string& modelHash, const std::string& mode, size_t capacity);
  bool get(const std::string& fen, float& value) const;
  void put(const std::string& fen, float value);
  double hitRate() const;
  size_t size() const;
 private:
  std::string model_;
  std::string mode_;
  size_t cap_;
  mutable size_t hits_ = 0;
  mutable size_t misses_ = 0;
  std::unordered_map<std::string, float> map_;
  std::vector<std::string> order_;
};
}
}
