import argparse
import json
import os
import subprocess
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import torch
from training.export.export import load_exported_arrays
from training.features.python_features import extract_features
from training.models.adaptive import AdaptiveModel
ARCH_MAP = [
    ("cw1", "cheap_head.fc1.weight"), ("cb1", "cheap_head.fc1.bias"),
    ("cwv", "cheap_head.fcv.weight"), ("cbv", "cheap_head.fcv.bias"),
    ("cww", "cheap_head.fcwdl.weight"), ("cbw", "cheap_head.fcwdl.bias"),
    ("dw", "difficulty.fc.weight"), ("db", "difficulty.fc.bias"),
    ("wq", "refine.wq.weight"), ("bq", "refine.wq.bias"),
    ("wk", "refine.wk.weight"), ("bk", "refine.wk.bias"),
    ("wvv", "refine.wv.weight"), ("bvv", "refine.wv.bias"),
    ("gabS", "refine.gab"),
    ("w1", "refined_head.fc1.weight"), ("b1", "refined_head.fc1.bias"),
    ("w2", "refined_head.fc2.weight"), ("b2", "refined_head.fc2.bias"),
    ("wvo", "refined_head.fcv.weight"), ("bvo", "refined_head.fcv.bias"),
    ("wwdl", "refined_head.fcwdl.weight"), ("bwdl", "refined_head.fcwdl.bias"),
]
def dequant(arr, header, name):
    if str(arr.dtype) in ("int8", "int16"):
        return arr.astype(np.float32) * header["scales"][name]
    return np.asarray(arr, dtype=np.float32)
def load_torch_adaptive(path):
    header, arrays = load_exported_arrays(path)
    m = AdaptiveModel(dim=header.get("token_dim", 16))
    sd = m.state_dict()
    for g in range(8):
        sd[f"embedder.tables.{g}.weight"].copy_(torch.from_numpy(dequant(arrays[f"emb{g}"], header, f"emb{g}")))
    for ek, pk in ARCH_MAP:
        sd[pk].copy_(torch.from_numpy(dequant(arrays[ek], header, ek).reshape(sd[pk].shape)))
    m.load_state_dict(sd)
    m.eval()
    return m, header
def batch_for(fen):
    feats = extract_features(fen)
    ids, masks = [], []
    for g in range(8):
        idx = [i for gg, i in feats if gg == g]
        ids.append(torch.tensor([idx if idx else [0]]))
        masks.append(torch.tensor([[1.0] * len(idx) if idx else [0.0]]))
    return ids, masks
def run_cpp(cpp_bin, model, fen, threshold):
    out = subprocess.check_output([cpp_bin, "--model", model, "--fen", fen, "--threshold", str(threshold)], text=True)
    p = out.strip().split()
    return {"value": float(p[0]), "wdl": [float(x) for x in p[1:4]], "difficulty": float(p[4]), "refined": int(p[5])}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--cpp-bin", default="build/rune_eval")
    ap.add_argument("--thresholds", default="0.0,0.1,0.2,0.3,0.5,0.7,0.9,1.0")
    ap.add_argument("--diff-tol", type=float, default=1e-5)
    ap.add_argument("--value-tol", type=float, default=1e-4)
    ap.add_argument("--time-iters", type=int, default=200)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",")]
    with open(args.positions) as f:
        fens = [l.strip() for l in f if l.strip()]
    model, header = load_torch_adaptive(args.model)
    model_threshold = float(header.get("threshold", 0.5))
    diffs_py = []
    diffs_cpp = []
    vals_py = []
    vals_cpp = []
    worst_diff = 0.0
    worst_val = 0.0
    ok = True
    for fen in fens:
        ids, masks = batch_for(fen)
        with torch.no_grad():
            out_v, out_w, mask = model.infer(ids, masks, mode="adaptive", threshold=model_threshold)
            _, _, _, diff_t = model.cheap_forward(ids, masks)
        d_py = float(diff_t.item())
        c = run_cpp(args.cpp_bin, args.model, fen, model_threshold)
        diffs_py.append(d_py)
        diffs_cpp.append(c["difficulty"])
        vals_py.append(float(out_v.item()))
        vals_cpp.append(c["value"])
        worst_diff = max(worst_diff, abs(d_py - c["difficulty"]))
        worst_val = max(worst_val, abs(float(out_v.item()) - c["value"]))
    print("difficulty max_abs %.3g (tol %.1g)" % (worst_diff, args.diff_tol))
    print("value max_abs %.3g (tol %.1g)" % (worst_val, args.value_tol))
    if worst_diff > args.diff_tol or worst_val > args.value_tol:
        ok = False
    sweep = []
    for t in thresholds:
        agrees = 0
        flips = []
        for i, fen in enumerate(fens):
            py_ref = diffs_py[i] >= t
            c = run_cpp(args.cpp_bin, args.model, fen, t)
            cpp_ref = bool(c["refined"])
            if py_ref == cpp_ref:
                agrees += 1
            else:
                flips.append({"fen": fen, "margin": abs(diffs_py[i] - t)})
        rate = sum(1 for i in range(len(fens)) if diffs_py[i] >= t) / max(1, len(fens))
        min_margin = min([f["margin"] for f in flips], default=float("inf"))
        noisy = all(f["margin"] < 1e-5 for f in flips)
        row = {"threshold": t, "refine_rate": rate, "agreement": agrees / len(fens),
               "flips": len(flips), "min_flip_margin": min_margin if flips else None,
               "pass": agrees == len(fens) or noisy}
        sweep.append(row)
        if not row["pass"]:
            ok = False
        print("t=%.2f rate=%.3f agree=%d/%d flips=%d min_margin=%s %s" % (
            t, rate, agrees, len(fens), len(flips),
            ("%.2g" % min_margin) if flips else "-",
            "PASS" if row["pass"] else "FAIL"))
    ids0, masks0 = batch_for(fens[0])
    with torch.no_grad():
        model.cheap_forward(ids0, masks0)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.time_iters):
            model.cheap_forward(ids0, masks0)
    t_cheap = (time.perf_counter() - t0) / args.time_iters * 1e6
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.time_iters):
            model(ids0, masks0)
    t_full = (time.perf_counter() - t0) / args.time_iters * 1e6
    print("torch cheap_us %.2f always_us %.2f" % (t_cheap, t_full))
    rep = {"model": args.model, "positions": len(fens), "difficulty_max_abs": worst_diff,
           "value_max_abs": worst_val, "sweep": sweep,
           "torch_cheap_us": t_cheap, "torch_always_us": t_full, "pass": ok}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
    sys.exit(0 if ok else 2)
if __name__ == "__main__":
    main()
