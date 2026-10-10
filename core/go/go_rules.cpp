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

#include "core/go/go_rules.h"

namespace rune {
namespace go {

int8_t goOpponent(int8_t stone) {
  if (stone == 1) return -1;
  return 1;
}

void goNeighbors(int sq, int n, std::vector<int>& out) {
  out.clear();
  int r = sq / n;
  int c = sq % n;
  if (r > 0) out.push_back((r - 1) * n + c);
  if (r + 1 < n) out.push_back((r + 1) * n + c);
  if (c > 0) out.push_back(r * n + (c - 1));
  if (c + 1 < n) out.push_back(r * n + (c + 1));
}

void goGroupAndLibs(const std::vector<int8_t>& board, int start, int n,
                    std::vector<int>& stones, std::vector<int>& libs) {
  stones.clear();
  libs.clear();
  int8_t target = board[static_cast<size_t>(start)];
  if (target == 0) return;
  std::vector<char> seen(static_cast<size_t>(n * n), 0);
  std::vector<char> present(static_cast<size_t>(n * n), 0);
  std::vector<int> stack;
  stack.push_back(start);
  seen[static_cast<size_t>(start)] = 1;
  std::vector<int> nb;
  while (!stack.empty()) {
    int cur = stack.back();
    stack.pop_back();
    stones.push_back(cur);
    goNeighbors(cur, n, nb);
    for (int v : nb) {
      int8_t bv = board[static_cast<size_t>(v)];
      if (bv == 0) {
        if (!present[static_cast<size_t>(v)]) {
          present[static_cast<size_t>(v)] = 1;
          libs.push_back(v);
        }
      } else if (bv == target && !seen[static_cast<size_t>(v)]) {
        seen[static_cast<size_t>(v)] = 1;
        stack.push_back(v);
      }
    }
  }
}

void goLibertyMap(const std::vector<int8_t>& board, int n, std::vector<int>& out) {
  out.assign(static_cast<size_t>(n * n), 0);
  std::vector<char> done(static_cast<size_t>(n * n), 0);
  std::vector<int> stones;
  std::vector<int> libs;
  for (int sq = 0; sq < n * n; ++sq) {
    if (board[static_cast<size_t>(sq)] == 0 || done[static_cast<size_t>(sq)]) continue;
    goGroupAndLibs(board, sq, n, stones, libs);
    int k = static_cast<int>(libs.size());
    for (int s : stones) {
      out[static_cast<size_t>(s)] = k;
      done[static_cast<size_t>(s)] = 1;
    }
  }
}

bool goPlayStone(const std::vector<int8_t>& board, int n, int sq, int8_t stone, bool hasKo,
                 int ko, std::vector<int8_t>& next, bool& outHasKo, int& outKo,
                 std::vector<int>& captured) {
  captured.clear();
  outHasKo = false;
  outKo = -1;
  if (board[static_cast<size_t>(sq)] != 0) return false;
  if (hasKo && sq == ko) return false;
  next = board;
  next[static_cast<size_t>(sq)] = stone;
  int8_t opp = goOpponent(stone);
  std::vector<char> seen(static_cast<size_t>(n * n), 0);
  std::vector<int> stones;
  std::vector<int> libs;
  std::vector<int> nb;
  goNeighbors(sq, n, nb);
  for (int adj : nb) {
    if (next[static_cast<size_t>(adj)] != opp || seen[static_cast<size_t>(adj)]) continue;
    goGroupAndLibs(next, adj, n, stones, libs);
    for (int s : stones) seen[static_cast<size_t>(s)] = 1;
    if (libs.empty()) {
      for (int s : stones) captured.push_back(s);
    }
  }
  for (int s : captured) next[static_cast<size_t>(s)] = 0;
  goGroupAndLibs(next, sq, n, stones, libs);
  if (libs.empty()) return false;
  if (captured.size() == 1) {
    goGroupAndLibs(next, sq, n, stones, libs);
    if (stones.size() == 1 && libs.size() == 1) {
      outHasKo = true;
      outKo = captured[0];
    }
  }
  return true;
}

void goLegalMoves(const std::vector<int8_t>& board, int n, uint8_t stm, bool hasKo, int ko,
                  std::vector<int>& out) {
  out.clear();
  int8_t stone = (stm == 0) ? 1 : -1;
  std::vector<int8_t> next;
  std::vector<int> captured;
  bool ok = false;
  int oko = -1;
  for (int sq = 0; sq < n * n; ++sq) {
    if (board[static_cast<size_t>(sq)] != 0) continue;
    if (hasKo && sq == ko) continue;
    if (goPlayStone(board, n, sq, stone, hasKo, ko, next, ok, oko, captured)) out.push_back(sq);
  }
  out.push_back(-1);
}

void goTerritory(const std::vector<int8_t>& board, int n, std::vector<int8_t>& owner) {
  owner.assign(static_cast<size_t>(n * n), 0);
  std::vector<char> done(static_cast<size_t>(n * n), 0);
  const int dr[4] = {-1, 1, 0, 0};
  const int dc[4] = {0, 0, -1, 1};
  for (int sq = 0; sq < n * n; ++sq) {
    if (board[static_cast<size_t>(sq)] != 0 || done[static_cast<size_t>(sq)]) continue;
    std::vector<int> region;
    std::vector<int8_t> border;
    std::vector<int> stack;
    stack.push_back(sq);
    done[static_cast<size_t>(sq)] = 1;
    while (!stack.empty()) {
      int cur = stack.back();
      stack.pop_back();
      region.push_back(cur);
      int r = cur / n;
      int c = cur % n;
      for (int k = 0; k < 4; ++k) {
        int nr = r + dr[k];
        int nc = c + dc[k];
        if (nr < 0 || nc < 0 || nr >= n || nc >= n) continue;
        int nsq = nr * n + nc;
        int8_t v = board[static_cast<size_t>(nsq)];
        if (v == 0 && !done[static_cast<size_t>(nsq)]) {
          done[static_cast<size_t>(nsq)] = 1;
          stack.push_back(nsq);
        } else if (v != 0) {
          bool found = false;
          for (int8_t b : border) {
            if (b == v) {
              found = true;
              break;
            }
          }
          if (!found) border.push_back(v);
        }
      }
    }
    int8_t fill = 0;
    if (border.size() == 1) fill = border[0];
    for (int s : region) owner[static_cast<size_t>(s)] = fill;
  }
}

float goAreaScore(const std::vector<int8_t>& board, int n, float komi) {
  int black = 0;
  int white = 0;
  for (int8_t v : board) {
    if (v == 1) ++black;
    else if (v == -1) ++white;
  }
  std::vector<int8_t> owner;
  goTerritory(board, n, owner);
  for (int sq = 0; sq < n * n; ++sq) {
    if (board[static_cast<size_t>(sq)] != 0) continue;
    if (owner[static_cast<size_t>(sq)] == 1) ++black;
    else if (owner[static_cast<size_t>(sq)] == -1) ++white;
  }
  return static_cast<float>(black - white) - komi;
}

}
}
