import argparse
import json
import os
import subprocess
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import torch
from training.export.export import load_exported_arrays
from training.features.python_features import extract_features
from training.models.dense import DenseModel
HEAD_MAP = [
    ("w1", "head.fc1.weight"), ("b1", "head.fc1.bias"),
    ("w2", "head.fc2.weight"), ("b2", "head.fc2.bias"),
    ("wvo", "head.fcv.weight"), ("bvo", "head.fcv.bias"),
    ("wwdl", "head.fcwdl.weight"), ("bwdl", "head.fcwdl.bias"),
]
def dequant(arr, scale, header, name):
    if str(arr.dtype) in ("int8", "int16"):
        return arr.astype(np.float32) * header["scales"][name]
    return np.asarray(arr, dtype=np.float32)
def load_torch_dense(path):
    header, arrays = load_exported_arrays(path)
    dims = header.get("token_dims", [32] * 8)
    m = DenseModel(variant=header.get("variant", "B"), token_dims=list(dims),
                   pooling=header.get("pooling", "none"),
                   pool_clip=bool(header.get("pool_clip", True)),
                   gate_on=bool(header.get("gate_on", False)))
    sd = m.state_dict()
    for g in range(8):
        sd[f"embedder.tables.{g}.weight"].copy_(
            torch.from_numpy(dequant(arrays[f"emb{g}"], None, header, f"emb{g}")))
    for ek, pk in HEAD_MAP:
        sd[pk].copy_(torch.from_numpy(dequant(arrays[ek], None, header, ek).reshape(sd[pk].shape)))
    m.load_state_dict(sd)
    m.eval()
    return m
def batch_for(fen):
    feats = extract_features(fen)
    ids, masks = [], []
    for g in range(8):
        idx = [i for gg, i in feats if gg == g]
        ids.append(torch.tensor([idx if idx else [0]]))
        masks.append(torch.tensor([[1.0] * len(idx) if idx else [0.0]]))
    return ids, masks
def run_cpp(cpp_bin, model, fen):
    out = subprocess.check_output([cpp_bin, "--model", model, "--fen", fen], text=True)
    parts = out.strip().split()
    return float(parts[0]), [float(x) for x in parts[1:4]]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--cpp-bin", default="build/rune_eval")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--quant-tol", type=float, default=0.01)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    with open(args.positions) as f:
        fens = [l.strip() for l in f if l.strip()]
    torch_models = {p: load_torch_dense(p) for p in args.models}
    rows = []
    ok = True
    base = None
    for fen in fens:
        ids, masks = batch_for(fen)
        vals = {}
        for p in args.models:
            with torch.no_grad():
                v, w = torch_models[p](ids, masks)
            vals[p] = float(v.item())
        cvals = {}
        for p in args.models:
            try:
                cvals[p] = run_cpp(args.cpp_bin, p, fen)[0]
            except Exception as e:
                print("cpp failed " + str(e))
                cvals[p] = float("nan")
        if base is None:
            base = args.models[0]
        row = {"fen": fen}
        for p in args.models:
            d_pc = abs(vals[p] - cvals[p])
            row[p] = {"py": vals[p], "cpp": cvals[p], "py_cpp": d_pc,
                      "pass": d_pc <= args.tol}
            if not row[p]["pass"]:
                ok = False
        d_q = abs(vals[args.models[0]] - vals[args.models[1]]) if len(args.models) > 1 else 0.0
        row["quant_gap"] = d_q
        if d_q > args.quant_tol:
            ok = False
        rows.append(row)
        print(fen[:44] + " " + " ".join("%.6f/%.6f" % (vals[p], cvals[p]) for p in args.models) + (" PASS" if all(rows[-1][p]["pass"] for p in args.models) and d_q <= args.quant_tol else " FAIL"))
    rep = {"models": args.models, "tol": args.tol, "quant_tol": args.quant_tol, "rows": rows, "pass": ok}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
    sys.exit(0 if ok else 2)
if __name__ == "__main__":
    main()
