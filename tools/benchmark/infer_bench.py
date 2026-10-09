import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

ARCHS = ["RUNE-MLP", "RUNE-ATTN", "RUNE-ATTN-GAB"]
REL_VARIANTS = [
    ("RUNE-REL-02", ["--tokens", "8", "--dim", "32", "--bias", "static"]),
    ("RUNE-REL-02", ["--tokens", "8", "--dim", "32", "--bias", "dynamic"]),
    ("RUNE-REL-02", ["--tokens", "6", "--dim", "32", "--bias", "dynamic"]),
    ("RUNE-REL-02", ["--tokens", "10", "--dim", "40", "--bias", "dynamic"]),
    ("RUNE-REL-02", ["--tokens", "8", "--dim", "24", "--bias", "static"]),
]


def run_stage_bench(build_dir, arch, extra=None):
    exe = os.path.join(build_dir, "rune_stage_bench")
    cmd = [exe, "--arch", arch] + (extra or [])
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    row = {}
    for line in out.strip().split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            row[k.strip()] = v.strip()
    return row


def torch_latency(arch, iters=100):
    import torch

    from training.models.rune_models import build_model

    model = build_model(arch)
    model.eval()
    ids = [torch.randint(0, 64, (1, 8)) for _ in range(9)]
    masks = [torch.ones(1, 8) for _ in range(9)]
    with torch.no_grad():
        for _ in range(10):
            model(ids, masks)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(ids, masks)
        t1 = time.perf_counter()
    return {"torch_forward_us": (t1 - t0) * 1e6 / iters, "params": model.parameter_count()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    table = {}
    for arch in ARCHS:
        cpp = run_stage_bench(args.build_dir, arch)
        py = torch_latency(arch)
        acc = sum(float(cpp[k]) for k in ("accumulator_refresh_us", "token_create_us"))
        mixer = sum(float(cpp[k]) for k in ("qkv_proj_us", "qkt_us", "gab_bias_act_us",
                                            "v_mix_us", "residual_us"))
        table[arch] = {
            "accumulator_cost_us": round(acc, 3),
            "attention_mixer_cost_us": round(mixer, 3) if "ATTN" in arch else 0.0,
            "head_cost_us": float(cpp["head_mlp_us"]),
            "full_eval_refresh_us": float(cpp["full_eval_refresh_us"]),
            "full_eval_incremental_us": float(cpp["full_eval_incremental_us"]),
            **py,
        }
    for arch, extra in REL_VARIANTS:
        cpp = run_stage_bench(args.build_dir, arch, extra)
        key = "REL-" + "-".join(extra[1::2]) + "-" + extra[-1]
        table[key] = {
            "rel_mixer_cost_us": float(cpp["rel_mixer_us"]),
            "rel_head_cost_us": float(cpp["rel_head_us"]),
            "rel_eval_refresh_us": float(cpp["rel_full_eval_refresh_us"]),
            "rel_eval_incremental_us": float(cpp["rel_full_eval_incremental_us"]),
            "rel_params": int(cpp["rel_params"]),
        }
    print(json.dumps(table, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(table, f, indent=2)


if __name__ == "__main__":
    main()
