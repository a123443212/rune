# RUNE — Relational Unified Neural Evaluator
# Copyright (C) 2026 a123443212
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# This project is dual-licensed under the MIT License and the
# Apache License, Version 2.0. You may choose either license
# when using, copying, modifying, or distributing this software.
#
# MIT License: https://opensource.org/license/mit
# Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, this
# software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied.

from training.games.chess import ChessGame
from training.games.go import GoGame
from training.games.shogi import ShogiGame
from training.games.xiangqi import XiangqiGame

_GAMES = {}


def register(game):
    _GAMES[game.game_id] = game
    return game


def get(game_id="chess"):
    if game_id not in _GAMES:
        raise ValueError(f"unknown game {game_id}")
    return _GAMES[game_id]


def available():
    return sorted(_GAMES)


register(ChessGame())
register(ShogiGame())
register(XiangqiGame())
register(GoGame())
