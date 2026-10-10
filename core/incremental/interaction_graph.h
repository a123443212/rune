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
#include <utility>
#include <vector>

namespace rune {
namespace v13 {

extern const char* kGraphVersion;
extern const char* kInvalidationVersion;

struct EdgeMeta {
  int source = 0;
  int target = 0;
};

class InteractionGraph {
 public:
  explicit InteractionGraph(int tokens);
  InteractionGraph(int tokens, const std::vector<std::pair<int, int>>& edges);

  int tokens() const { return tokens_; }
  int numEdges() const { return static_cast<int>(edges_.size()); }
  bool isDense() const { return static_cast<int>(edges_.size()) == tokens_ * tokens_; }
  const char* version() const { return kGraphVersion; }

  std::vector<std::pair<int, int>> affectedEdges(const std::vector<int>& changed) const;
  std::vector<int> affectedRows(const std::vector<int>& changed) const;
  EdgeMeta edgeMeta(int a, int b) const;
  InteractionGraph pruned(const std::vector<std::pair<int, int>>& keep) const;

 private:
  int tokens_ = 8;
  std::vector<std::pair<int, int>> edges_;
};

InteractionGraph buildDenseGraph(int tokens);

}
}
