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

from training.datasets.pipeline import phase_of_record
from training.features.python_features import extract_features, parse_fen


def material_bucket(fen):
    board, _, _, _ = parse_fen(fen)
    non_pawn = sum(1 for c in board if c is not None and c[0] not in ("p", "k"))
    if non_pawn >= 12:
        return "heavy"
    if non_pawn >= 6:
        return "medium"
    return "light"


def tactical_proxy(fen):
    threats = sum(1 for g, _ in extract_features(fen) if g == 5)
    return "tactical" if threats >= 4 else "quiet"


def composition(records):
    phases = {"opening": 0, "middlegame": 0, "endgame": 0}
    names = ["opening", "middlegame", "endgame"]
    materials = {"heavy": 0, "medium": 0, "light": 0}
    tactical = {"tactical": 0, "quiet": 0}
    for r in records:
        phases[names[phase_of_record(r)]] += 1
        materials[material_bucket(r["fen"])] += 1
        tactical[tactical_proxy(r["fen"])] += 1
    n = max(1, len(records))
    return {
        "n": len(records),
        "phase": {k: v / n for k, v in phases.items()},
        "material": {k: v / n for k, v in materials.items()},
        "tactical_proxy": {k: v / n for k, v in tactical.items()},
    }
