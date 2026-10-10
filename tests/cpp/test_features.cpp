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

#include <random>
#include <vector>

#include "core/accumulators/grouped_accumulator.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"
#include "tests/cpp/test_framework.h"

using namespace rune;

void testFeatures() {
  Board b;
  std::vector<ActiveFeature> f1;
  GroupedFeatureSet::extract(b, f1);
  CHECK(!f1.empty());
  for (size_t i = 1; i < f1.size(); ++i) CHECK(f1[i - 1] < f1[i]);
  for (const ActiveFeature& f : f1) CHECK(f.index < GroupedFeatureSet::vocabSize(f.group));
  std::vector<ActiveFeature> f2;
  GroupedFeatureSet::extract(b, f2);
  CHECK(f1 == f2);
  std::vector<ActiveFeature> added, removed;
  GroupedFeatureSet::diffFeatures(f1, f2, added, removed);
  CHECK(added.empty() && removed.empty());
  std::vector<Move> moves;
  b.generateLegalMoves(moves);
  CHECK(b.makeMove(moves[0]));
  std::vector<ActiveFeature> f3;
  GroupedFeatureSet::extract(b, f3);
  GroupedFeatureSet::diffFeatures(f1, f3, added, removed);
  CHECK(!added.empty() || !removed.empty());
  b.unmakeMove();
}

static void checkAccumulatorMatches(const GroupedAccumulator& inc, GroupedAccumulator& ref,
                                    const std::vector<ActiveFeature>& cur) {
  ref.refresh(cur);
  float ti[256];
  float tr[256];
  inc.tokens(ti);
  ref.tokens(tr);
  for (int i = 0; i < 256; ++i) CHECK_CLOSE(ti[i], tr[i], 1e-5);
}

void testAccumulatorIncremental() {
  std::mt19937 rng(999);
  EmbeddingTables tables;
  tables.init(42);
  const char* fens[] = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"};
  for (const char* fen : fens) {
    Board b(fen);
    GroupedAccumulator inc(&tables);
    GroupedAccumulator ref(&tables);
    std::vector<ActiveFeature> prev;
    GroupedFeatureSet::extract(b, prev);
    inc.refresh(prev);
    for (int step = 0; step < 80; ++step) {
      std::vector<Move> moves;
      b.generateLegalMoves(moves);
      if (moves.empty()) break;
      Move m = moves[rng() % moves.size()];
      CHECK(b.makeMove(m));
      std::vector<ActiveFeature> cur;
      GroupedFeatureSet::extract(b, cur);
      std::vector<ActiveFeature> added, removed;
      GroupedFeatureSet::diffFeatures(prev, cur, added, removed);
      inc.applyDiff(added, removed);
      checkAccumulatorMatches(inc, ref, cur);
      prev = cur;
    }
    for (int step = 0; step < 40; ++step) {
      if (b.historySize() == 0) break;
      b.unmakeMove();
      std::vector<ActiveFeature> cur;
      GroupedFeatureSet::extract(b, cur);
      std::vector<ActiveFeature> added, removed;
      GroupedFeatureSet::diffFeatures(prev, cur, added, removed);
      inc.applyDiff(added, removed);
      checkAccumulatorMatches(inc, ref, cur);
      prev = cur;
    }
  }
}
