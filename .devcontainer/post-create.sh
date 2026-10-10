#!/usr/bin/env bash
set -e
pip install -r requirements.txt
cmake -S . -B build -DRUNE_BUILD_BINDINGS=OFF
cmake --build build -j4 --target rune_tests rune_eval
