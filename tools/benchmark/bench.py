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

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def bench_torch(arch_id, iters=200):
    import torch

    from training.models.rune_models import build_model

    model = build_model(arch_id)
    model.eval()
    ids = [torch.randint(0, 64, (1, 8)) for _ in range(9)]
    masks = [torch.ones(1, 8) for _ in range(9)]
    with torch.no_grad():
        for _ in range(20):
            model(ids, masks)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(ids, masks)
        t1 = time.perf_counter()
    return {"torch_forward_us": (t1 - t0) * 1e6 / iters, "params": model.parameter_count()}


def bench_cpp(build_dir):
    import subprocess

    exe = os.path.join(build_dir, "rune_bench")
    if not os.path.exists(exe):
        return {"cpp_bench": "not built"}
    out = subprocess.run([exe], capture_output=True, text=True).stdout
    row = {}
    for line in out.strip().split("\n"):
        parts = line.split(":")
        if len(parts) == 2:
            row[parts[0].strip()] = parts[1].strip()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="RUNE-ATTN-GAB")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    result = {"arch": args.arch}
    result.update(bench_torch(args.arch, args.iters))
    result.update(bench_cpp(args.build_dir))
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
