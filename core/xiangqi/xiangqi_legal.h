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
#include <vector>

#include "core/xiangqi/xiangqi_board.h"

namespace rune {
namespace xiangqi {

std::string xiangqiEncodeMove(int fr, int to);
bool xiangqiDecodeMove(const std::string& mv, int& fr, int& to);
bool xiangqiKingInCheck(const XiangqiBoard& board, uint8_t color);
bool xiangqiLegalMoves(const std::string& state, std::vector<std::string>& out);
bool xiangqiApplyMove(const std::string& state, const std::string& mv, std::string& out);
std::string xiangqiGameResult(const std::string& state);

}
}
