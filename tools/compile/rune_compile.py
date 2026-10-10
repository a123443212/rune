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
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from training.compiler.artifact import write_compiled
from training.compiler.graph import export_canonical_graph
from training.compiler.ir import build_ir, verify_ir
from training.compiler.memory import plan_memory
from training.compiler.packing import pack_weights
from training.compiler.plan import select_kernels
from training.compiler.precomputed import build_precomputed
from training.export.export import load_exported_arrays


def parse_target(s):
    parts = s.split(",")
    tgt = {"cpu": "generic-x86-64", "isa": "portable", "vector_width": 1}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k in ("cpu", "isa"):
                tgt[k] = v
            elif k == "vector_width":
                tgt[k] = int(v)
    if tgt["isa"] == "avx2":
        tgt["vector_width"] = 8
    if tgt["isa"] == "avx512":
        tgt["vector_width"] = 16
    return tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="isa=portable")
    ap.add_argument("--graph-out", default="")
    ap.add_argument("--cache-dir", default="")
    args = ap.parse_args()
    header, arrays = load_exported_arrays(args.model)
    spec = {
        "arch": header.get("architecture_id", header.get("arch", "")),
        "arch_version": header.get("architecture_version", header.get("arch_version", "0.2.0")),
        "tokens": header.get("tokens", 8),
        "token_dim": header.get("token_dim", 32),
        "quantization": header.get("quantization", "fp32"),
        "gate": header.get("gate", "clip"),
        "alpha": header.get("alpha", 1.0),
        "head_h1": header.get("head_h1", 128),
        "head_h2": header.get("head_h2", 32),
        "threshold": header.get("threshold", 0.5),
        "t_high": header.get("t_high", header.get("threshold", 0.5)),
    }
    if header.get("t_low") is not None:
        spec["t_low"] = header.get("t_low")
    tlist = header.get("tensor_metadata", header.get("tensors", []))
    target = parse_target(args.target)
    ir = build_ir(spec, tlist, target)
    errs = verify_ir(ir)
    if errs:
        print("ir invalid: %s" % "; ".join(errs))
        return 2
    plan, fusion = select_kernels(ir)
    ir["kernel_plan"] = plan
    ir["fusion"] = fusion
    ir["memory"] = plan_memory(ir)
    scales = {}
    try:
        qm = header.get("quantization_metadata", {}).get("scales", {})
        scales.update(qm)
        scales.update(header.get("scales", {}))
    except Exception:
        pass
    packed, packing_meta = pack_weights(arrays, tlist, target["isa"])
    consts = build_precomputed(spec, arrays, scales)
    precomputed = {"packing_meta": packing_meta, "constants": consts}
    order = [t.get("name") for t in tlist]
    info = write_compiled(args.out, header, packed, order, ir, precomputed)
    if args.graph_out:
        export_canonical_graph(ir, args.graph_out)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        cpath = os.path.join(args.cache_dir, "rune-%s.json" % info["cache_key"])
        with open(cpath, "w") as f:
            json.dump({"source_hash": info["source_hash"], "plan_hash": info["plan_hash"], "target": target, "model": ir["model"]}, f, indent=2, sort_keys=True)
    print("compiled %s isa %s bytes %d plan %s cache %s" % (args.out, target["isa"], info["bytes"], info["plan_hash"], info["cache_key"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
