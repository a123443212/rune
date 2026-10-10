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

#include <cstdint>
#include <vector>

namespace rune {
namespace go {

int8_t goOpponent(int8_t stone);
void goNeighbors(int sq, int n, std::vector<int>& out);
void goGroupAndLibs(const std::vector<int8_t>& board, int start, int n,
                    std::vector<int>& stones, std::vector<int>& libs);
void goLibertyMap(const std::vector<int8_t>& board, int n, std::vector<int>& out);
bool goPlayStone(const std::vector<int8_t>& board, int n, int sq, int8_t stone, bool hasKo,
                 int ko, std::vector<int8_t>& next, bool& outHasKo, int& outKo,
                 std::vector<int>& captured);
void goLegalMoves(const std::vector<int8_t>& board, int n, uint8_t stm, bool hasKo, int ko,
                  std::vector<int>& out);
void goTerritory(const std::vector<int8_t>& board, int n, std::vector<int8_t>& owner);
float goAreaScore(const std::vector<int8_t>& board, int n, float komi);

}
}
