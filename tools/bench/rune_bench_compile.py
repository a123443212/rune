import argparse
import json
import os
import subprocess
import sys


def run(cmd):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


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
    ap.add_argument("--compiled", default="")
    ap.add_argument("--cpp-bench", default="build/rune_bench_compile")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cpp_out = run([args.cpp_bench, "--iters", str(args.iters)])
    cpp = parse_kv(cpp_out)
    rep = {"model": args.model, "compiled": args.compiled, "cpp_compile_bench": cpp, "cpp_raw": cpp_out}
    if args.compiled:
        info_out = run(["build/rune_compile_support", "--model", args.compiled])
        rep["artifact"] = info_out
        print(info_out)
    print("qkv_fused_us %.3f" % cpp.get("qkv_fused_us", float("nan")))
    print("score_bias_gate_us %.3f" % cpp.get("score_bias_gate_us", float("nan")))
    print("mix_residual_us %.3f" % cpp.get("mix_residual_us", float("nan")))
    print("mixer_generic_us %.3f" % cpp.get("mixer_generic_us", float("nan")))
    print("head_generic_us %.3f" % cpp.get("head_generic_us", float("nan")))
    if "qkv_fused_us" in cpp and "mixer_generic_us" in cpp and cpp["mixer_generic_us"] > 0:
        fused = cpp["qkv_fused_us"] + cpp["score_bias_gate_us"] + cpp["mix_residual_us"]
        print("fused_total_us %.3f generic_mixer_head_us %.3f speedup %.2fx" % (fused, cpp["mixer_generic_us"] + cpp["head_generic_us"], (cpp["mixer_generic_us"] + cpp["head_generic_us"]) / fused if fused else float("nan")))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
