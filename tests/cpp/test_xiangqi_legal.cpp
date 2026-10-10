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

#include <string>
#include <vector>

#include "core/xiangqi/xiangqi_legal.h"
#include "tests/cpp/test_framework.h"

using namespace rune;
using namespace rune::xiangqi;

namespace {

bool hasMove(const std::vector<std::string>& moves, const std::string& mv) {
  for (const auto& m : moves) {
    if (m == mv) return true;
  }
  return false;
}

std::string startFen() {
  return "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1";
}

void testStartposCounts() {
  std::vector<std::string> red;
  CHECK(xiangqiLegalMoves(startFen(), red));
  CHECK(red.size() == 44);
  for (size_t i = 1; i < red.size(); ++i) CHECK(red[i - 1] < red[i]);
  std::string nxt;
  CHECK(xiangqiApplyMove(startFen(), red[0], nxt));
  std::vector<std::string> black;
  CHECK(xiangqiLegalMoves(nxt, black));
  CHECK(black.size() == 44);
}

void testApplyRejects() {
  std::string out;
  CHECK(!xiangqiApplyMove(startFen(), "9999", out));
  std::vector<std::string> red;
  CHECK(xiangqiLegalMoves(startFen(), red));
  CHECK(xiangqiApplyMove(startFen(), red[0], out));
  CHECK(out != startFen());
}

void testMateIsLoss() {
  std::vector<std::string> moves;
  CHECK(xiangqiLegalMoves("3aka3/2H1P4/9/4R4/9/9/9/9/9/4K4 b - - 0 1", moves));
  CHECK(moves.empty());
  CHECK(xiangqiGameResult("3aka3/2H1P4/9/4R4/9/9/9/9/9/4K4 b - - 0 1") == "1-0");
  CHECK(xiangqiGameResult(startFen()) == "*");
}

void testHorseLeg() {
  std::vector<std::string> moves;
  CHECK(xiangqiLegalMoves("3k5/9/9/9/3p5/3H5/9/9/9/5K3 w - - 0 1", moves));
  CHECK(!hasMove(moves, "3958"));
  CHECK(!hasMove(moves, "3956"));
  CHECK(hasMove(moves, "3950"));
}

}

void testXiangqiLegal() {
  testStartposCounts();
  testApplyRejects();
  testMateIsLoss();
  testHorseLeg();
}
