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

#include "core/incremental/interaction_graph.h"

#include <algorithm>

namespace rune {
namespace v13 {

const char* kGraphVersion = "v13-graph-01";
const char* kInvalidationVersion = "v13-inv-01";

InteractionGraph::InteractionGraph(int tokens) : tokens_(tokens) {
  for (int a = 0; a < tokens_; ++a) {
    for (int b = 0; b < tokens_; ++b) edges_.emplace_back(a, b);
  }
}

InteractionGraph::InteractionGraph(int tokens,
                                   const std::vector<std::pair<int, int>>& edges)
    : tokens_(tokens), edges_(edges) {}

std::vector<std::pair<int, int>> InteractionGraph::affectedEdges(
    const std::vector<int>& changed) const {
  std::vector<int> mark(static_cast<size_t>(tokens_), 0);
  for (int t : changed) {
    if (t >= 0 && t < tokens_) mark[static_cast<size_t>(t)] = 1;
  }
  std::vector<std::pair<int, int>> out;
  for (const auto& e : edges_) {
    if (mark[static_cast<size_t>(e.first)] || mark[static_cast<size_t>(e.second)]) {
      out.push_back(e);
    }
  }
  return out;
}

std::vector<int> InteractionGraph::affectedRows(const std::vector<int>& changed) const {
  std::vector<int> mark(static_cast<size_t>(tokens_), 0);
  for (const auto& e : affectedEdges(changed)) {
    mark[static_cast<size_t>(e.first)] = 1;
  }
  std::vector<int> out;
  for (int a = 0; a < tokens_; ++a) {
    if (mark[static_cast<size_t>(a)]) out.push_back(a);
  }
  return out;
}

EdgeMeta InteractionGraph::edgeMeta(int a, int b) const {
  EdgeMeta m;
  m.source = a;
  m.target = b;
  return m;
}

InteractionGraph InteractionGraph::pruned(
    const std::vector<std::pair<int, int>>& keep) const {
  return InteractionGraph(tokens_, keep);
}

InteractionGraph buildDenseGraph(int tokens) { return InteractionGraph(tokens); }

}
}
