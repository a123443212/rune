import argparse
import random
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from training.compiler.ir import build_ir, verify_ir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    passed = 0
    rejected = 0
    for t in range(args.trials):
        tokens = rng.choice([0, 6, 8, 10, 16, 64])
        dim = rng.choice([0, 24, 32, 40, 128, 256])
        quant = rng.choice(["fp32", "int8", "int16", "int4", "fp16", ""])
        isa = rng.choice(["portable", "avx2", "avx512", "neon", "", "cuda"])
        spec = {"arch": "RUNE-ATTN-GAB", "tokens": tokens, "token_dim": dim, "quantization": quant}
        try:
            ir = build_ir(spec, [], {"cpu": "fuzz", "isa": isa, "vector_width": 1})
            errs = verify_ir(ir)
        except Exception as e:
            print("trial %d exception %s" % (t, e))
            return 2
        valid = tokens in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16) and dim in (1, 8, 16, 24, 32, 40, 64, 128) and quant in ("fp32", "int8", "int16") and isa in ("portable", "avx2", "avx512")
        if valid and errs:
            print("trial %d FALSE REJECT tokens=%s dim=%s quant=%s isa=%s errs=%s" % (t, tokens, dim, quant, isa, errs))
            return 2
        if not valid and not errs:
            print("trial %d SILENT ACCEPT tokens=%s dim=%s quant=%s isa=%s" % (t, tokens, dim, quant, isa))
            return 2
        if errs:
            rejected += 1
        else:
            passed += 1
    print("fuzz %d trials passed=%d rejected=%d" % (args.trials, passed, rejected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
