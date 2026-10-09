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
