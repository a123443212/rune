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

#include <cstdio>
#include <string>
#include "core/board/board.h"
#include "core/engine/search.h"
int main(int argc, char** argv) {
  std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  int depth = 3;
  for (int i = 1; i + 1 < argc; ++i) {
    std::string a = argv[i];
    if (a == "--fen") fen = argv[i + 1];
    if (a == "--depth") depth = std::atoi(argv[i + 1]);
  }
  rune::Board b(fen);
  rune::eng::LazyConfig cfg;
  rune::eng::EvalFn eva = [](rune::Board& x) -> float {
    std::vector<rune::Move> lm;
    x.generateLegalMoves(lm);
    return (float)((int)lm.size() % 7) * 0.05f;
  };
  rune::eng::EvalFn evb = [](rune::Board& x) -> float {
    std::vector<rune::Move> lm;
    x.generateLegalMoves(lm);
    return (float)(((int)lm.size() + 1) % 7) * 0.05f;
  };
  rune::Board b1(fen);
  rune::Board b2(fen);
  rune::eng::SearchStats s1;
  rune::eng::SearchStats s2;
  float v1 = rune::eng::searchRoot(b1, depth, eva, cfg, s1, -1e9f, 1e9f);
  float v2 = rune::eng::searchRoot(b2, depth, evb, cfg, s2, -1e9f, 1e9f);
  std::printf("a_score %.4f a_nodes %ld\n", v1, s1.nodes);
  std::printf("b_score %.4f b_nodes %ld\n", v2, s2.nodes);
  if (v1 == v2) {
    std::printf("status PARITY\n");
    return 0;
  }
  std::printf("first_divergence ply 0 fen %s a %.4f b %.4f\n", fen.c_str(), v1, v2);
  std::printf("status DIVERGED\n");
  return 2;
}
