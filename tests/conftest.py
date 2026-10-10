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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BUILD_DIRS = [
    os.path.join(os.path.dirname(__file__), "..", "build"),
]


def pytest_addoption(parser):
    pass


def find_binding():
    for d in BUILD_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for name in files:
                if name.startswith("rune_bindings") and name.endswith((".so", ".pyd", ".dylib")):
                    return root
    return None
