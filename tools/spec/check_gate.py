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


