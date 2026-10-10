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

import os

import sys


ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
def read(path):
    out = open(path).read()
    return out


SPEC = read(os.path.join(ROOT, "spec", "numerical-contract.md"))

PY = read(os.path.join(ROOT, "training", "models", "multi_head.py"))
RS = read(os.path.join(ROOT, "crates", "rune-kernel", "src", "lib.rs"))
CPP = read(os.path.join(ROOT, "core", "architectures", "base", "architecture.cpp"))

want = ["screlu", "clip", "hard_sigmoid"]
miss = []
for w in want:
    hit = (w in SPEC) and (w in PY) and (w in RS) and (w in CPP)
    print(w + " " + ("ok" if hit else "missing"))
    miss = miss + ([] if hit else [w])
sys.exit(1 if miss else 0)


