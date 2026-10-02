import glob
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
        cands = glob.glob(os.path.join(d, "rune_bindings*.so"))
        if cands:
            return os.path.dirname(cands[0])
    return None
