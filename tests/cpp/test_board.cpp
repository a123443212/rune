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
#include <string>
#include <vector>

#include "core/board/board.h"
#include "core/position/position.h"
#include "tests/cpp/test_framework.h"

using namespace rune;

static uint64_t perft(Board& b, int depth) {
  if (depth == 0) return 1;
  std::vector<Move> moves;
  b.generatePseudoLegalMoves(moves);
  uint64_t nodes = 0;
  for (const Move& m : moves) {
    if (b.makeMove(m)) {
      nodes += perft(b, depth - 1);
      b.unmakeMove();
    }
  }
  return nodes;
}

void testBoard() {
  Board b;
  CHECK(b.toFen() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  std::vector<Move> moves;
  b.generateLegalMoves(moves);
  CHECK(moves.size() == 20);
  CHECK(perft(b, 1) == 20);
  CHECK(perft(b, 2) == 400);
  CHECK(perft(b, 3) == 8902);
  Board kiwi("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
  CHECK(perft(kiwi, 1) == 48);
  CHECK(perft(kiwi, 2) == 2039);
  Board bad("8/8/8/8/8/8/8/8 w - - 0 1");
  CHECK(!bad.isLegalPosition());
  Board pawnFirst("P7/8/8/8/8/8/8/K6k w - - 0 1");
  CHECK(!pawnFirst.isLegalPosition());
  Position p("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
  CHECK(p.board().sideToMove() == Color::Black);
  CHECK(p.board().epSquare() == makeSq(4, 2));
  CHECK(p.meta().phase == 0);
}

void testMakeUnmakeConsistency() {
  std::mt19937 rng(1234);
  const char* fens[] = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
      "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"};
  for (const char* fen : fens) {
    Board b(fen);
    for (int iter = 0; iter < 60; ++iter) {
      std::string beforeFen = b.toFen();
      uint64_t beforeHash = b.hashKey();
      std::vector<Move> moves;
      b.generateLegalMoves(moves);
      if (moves.empty()) break;
      Move m = moves[rng() % moves.size()];
      CHECK(b.makeMove(m));
      b.unmakeMove();
      CHECK(b.toFen() == beforeFen);
      CHECK(b.hashKey() == beforeHash);
    }
  }
}

void testKingTwoSquareOffRank() {
  Board b("7k/8/8/8/8/8/K7/7R w - - 0 1");
  std::string before = b.toFen();
  Move m;
  m.from = 8;
  m.to = 6;
  CHECK(b.makeMove(m));
  b.unmakeMove();
  CHECK(b.toFen() == before);
}

void testEnPassantUnmake() {
  Board b("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1");
  std::string before = b.toFen();
  Move m;
  m.from = makeSq(4, 4);
  m.to = makeSq(3, 5);
  CHECK(b.makeMove(m));
  b.unmakeMove();
  CHECK(b.toFen() == before);
}

void testHashKeyEp() {
  Board a("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
  Board b("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
  CHECK(a.hashKey() != b.hashKey());
}
