import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from training.datasets import pipeline as P
from training.datasets.siblings import build_sibling_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--max-pairs", type=int, default=2000)
    ap.add_argument("--margin-min", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build-dir", default="build")
    args = ap.parse_args()

    sys.path.insert(0, args.build_dir)
    import rune_bindings as rb

    from training.export.export import load_exported_arrays
    from training.models.rune_models import EXPORT_ORDER

    header, arrays = load_exported_arrays(args.teacher)
    arch = header["arch"]
    if arch == "RUNE-REL-02":
        model = rb.FlexModel(header["tokens"], header["token_dim"], header.get("gate", "clip"),
                             header.get("alpha", 1.0), header["geometric_bias"] == "dynamic")
    else:
        model = rb.RuneModel(arch)
    for g in range(8):
        arr = arrays[f"emb{g}"]
        if str(arr.dtype) == "int8":
            arr = arr.astype("float32") * header["scales"][f"emb{g}"]
        elif str(arr.dtype) == "int16":
            arr = arr.astype("float32") * header["scales"][f"emb{g}"]
        model.set_embedding(g, [float(x) for x in arr.reshape(-1)])
    names = list(EXPORT_ORDER[arch]) if arch != "RUNE-REL-02" else None
    if names is None:
        names = ["wq", "bq", "wk", "bk", "wvv", "bvv", "gabS"]
        if header["geometric_bias"] == "dynamic":
            names += ["dynU", "dynW"]
        names += ["w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"]
    flat = []
    for n in names:
        flat.extend([float(x) for x in arrays[n].reshape(-1)])
    model.set_arch_tensors(names, flat)

    def teacher_fn(fen):
        return model.eval_fen(fen)[0]

    pool = P.load_jsonl(args.pool)
    parents = sorted({r["fen"] for r in pool})
    pairs = build_sibling_pairs(rb, parents, teacher_fn, teacher_id=args.teacher,
                                max_pairs=args.max_pairs, margin_min=args.margin_min,
                                seed=args.seed)
    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(json.dumps({"parents": len(parents), "pairs": len(pairs),
                      "teacher": args.teacher}, indent=2))


if __name__ == "__main__":
    main()
