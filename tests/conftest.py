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
