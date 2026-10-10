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

import sys
sys.path.insert(0, ".")
from training.datasets.pipeline import load_jsonl
from training.datasets.pipeline import split_by_game
pool = load_jsonl(sys.argv[1])
kept = [r for r in pool if "value" in r]
parts = split_by_game(kept, seed=42)
print("train " + str(len(parts["train"])))
print("val " + str(len(parts["val"])))
print("test " + str(len(parts["test"])))

