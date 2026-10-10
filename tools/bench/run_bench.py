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
import json
import os
import subprocess
import sys
def run(cmd):
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    return out
def parse_kv(out):
    d = {}
    for line in out.splitlines():
        p = line.strip().split()
        if len(p) >= 2:
            try:
                d[p[0]] = float(p[1])
            except Exception:
                pass
    return d
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cpp-bench", default="build/rune_bench_v10")
    ap.add_argument("--cpp-bench-avx2", default="")
    ap.add_argument("--rust-bench", default="target/debug/rune")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cpp_out = run([args.cpp_bench, "--threads", "1", "--path", "scalar"])
    cpp = parse_kv(cpp_out)
    cpp_avx2 = {}
    cpp_avx2_raw = ""
    if args.cpp_bench_avx2:
        cpp_avx2_raw = run([args.cpp_bench_avx2, "--threads", "1", "--path", "avx2"])
        cpp_avx2 = parse_kv(cpp_avx2_raw)
    rust_scalar_out = run([args.rust_bench, "bench", "--model", args.model, "--iters", str(args.iters), "--kernel", "scalar"])
    rust_scalar = parse_kv(rust_scalar_out)
    rust_simd_out = run([args.rust_bench, "bench", "--model", args.model, "--iters", str(args.iters), "--kernel", "simd"])
    rust_simd = parse_kv(rust_simd_out)
    rep = {"model": args.model, "cpp_scalar": cpp, "cpp_avx2": cpp_avx2,
           "rust_scalar": rust_scalar, "rust_simd": rust_simd,
           "cpp_raw": cpp_out, "cpp_avx2_raw": cpp_avx2_raw,
           "rust_scalar_raw": rust_scalar_out, "rust_simd_raw": rust_simd_out}
    print("cpp_scalar " + json.dumps(cpp, indent=2))
    if cpp_avx2:
        print("cpp_avx2 " + json.dumps(cpp_avx2, indent=2))
    print("rust_scalar " + json.dumps(rust_scalar, indent=2))
    print("rust_simd " + json.dumps(rust_simd, indent=2))
    if "us_per_eval" in rust_scalar and "full_eval_refresh_us" in cpp:
        r = rust_scalar["us_per_eval"]
        c = cpp["full_eval_refresh_us"]
        print("scalar rust_us %.3f cpp_us %.3f ratio %.3f" % (r, c, r / c if c else float("nan")))
    if "us_per_eval" in rust_simd and cpp_avx2.get("full_eval_refresh_us"):
        r = rust_simd["us_per_eval"]
        c = cpp_avx2["full_eval_refresh_us"]
        print("simd rust_us %.3f cpp_us %.3f ratio %.3f" % (r, c, r / c if c else float("nan")))
    if "us_per_eval" in rust_scalar and "us_per_eval" in rust_simd:
        print("rust simd speedup %.2fx" % (rust_scalar["us_per_eval"] / rust_simd["us_per_eval"]))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
if __name__ == "__main__":
    main()
