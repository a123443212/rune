import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbqkb1r/pp2pppp/5n2/2pp4/3P4/2N5/PPP1PPPP/R1BQKBNR w KQkq - 0 1",
]


def build_torch_model(arch, rel_params, seed):
    torch.manual_seed(seed)
    if arch == "RUNE-REL-02":
        from training.models.relational import build_rel_model

        return build_rel_model(**rel_params)
    from training.models.rune_models import build_model

    return build_model(arch)


def torch_evals(model, arch, fens):
    from training.datasets.flex_dataset import make_flex_loader
    from training.datasets.rune_dataset import make_loader

    recs = [{"fen": f, "value": 0.0, "wdl": 1} for f in fens]
    flex = arch == "RUNE-REL-02"
    loader, _ = (make_flex_loader if flex else make_loader)(recs, batch_size=8, shuffle=False)
    model.eval()
    vals, wdls = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:
                ids, masks, ctx, _, _ = batch
                v, w = model(ids, masks, ctx)
            else:
                ids, masks, _, _ = batch
                v, w = model(ids, masks)
            vals.extend(v.tolist())
            wdls.extend(w.argmax(dim=-1).tolist())
    return vals, wdls


def load_cpp_model(build_dir, path, header):
    sys.path.insert(0, build_dir)
    import rune_bindings as rb

    from training.export.export import load_exported_arrays

    header, arrays = load_exported_arrays(path)
    arch = header["arch"]
    if arch == "RUNE-REL-02":
        model = rb.FlexModel(header["tokens"], header["token_dim"], header.get("gate", "clip"),
                             header.get("alpha", 1.0), header["geometric_bias"] == "dynamic")
    else:
        model = rb.RuneModel(arch)
    for g in range(9):
        arr = arrays[f"emb{g}"]
        if str(arr.dtype) in ("int8", "int16"):
            arr = arr.astype("float32") * header["scales"][f"emb{g}"]
        model.set_embedding(g, [float(x) for x in arr.reshape(-1)])
    names = [t["name"] for t in header["tensors"] if not t["name"].startswith("emb")]
    flat = []
    for n in names:
        flat.extend([float(x) for x in arrays[n].reshape(-1)])
    model.set_arch_tensors(names, flat)
    return model


def compare(ref_vals, ref_wdls, vals, wdls):
    dv = [abs(a - b) for a, b in zip(ref_vals, vals)]
    same_sign = sum(1 for a, b in zip(ref_vals, vals) if (a >= 0) == (b >= 0))
    same_wdl = sum(1 for a, b in zip(ref_wdls, wdls) if a == b)
    n = max(1, len(vals))
    return {
        "max_abs_diff": max(dv) if dv else 0.0,
        "mean_abs_diff": sum(dv) / n,
        "sign_consistency": same_sign / n,
        "wdl_consistency": same_wdl / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="RUNE-MLP")
    ap.add_argument("--rel-params", default="{}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--out", default="")
    ap.add_argument("--workdir", default="/tmp/rune_quant")
    args = ap.parse_args()

    from training.export.export import export_model

    os.makedirs(args.workdir, exist_ok=True)
    rel_params = json.loads(args.rel_params)
    model = build_torch_model(args.arch, rel_params, args.seed)
    ref_vals, ref_wdls = torch_evals(model, args.arch, FENS)
    report = {"arch": args.arch, "positions": len(FENS)}
    for quant in ("fp32", "int16", "int8"):
        path = os.path.join(args.workdir, f"m_{quant}.rune")
        export_model(model, path, quantization=quant)
        size = os.path.getsize(path)
        cpp = load_cpp_model(args.build_dir, path, None)
        vals = [cpp.eval_fen(f)[0] for f in FENS]
        wdls = [int(np.argmax(cpp.eval_fen(f)[1])) for f in FENS]
        report[quant] = {**compare(ref_vals, ref_wdls, vals, wdls), "bytes": size}
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
