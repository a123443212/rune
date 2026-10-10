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
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from training.datasets import pipeline as P
from training.datasets.rune_dataset import make_loader


def build_teacher_model(arch, params):
    if arch.startswith("RUNE-05"):
        from training.models.uncertainty import build_search_model

        return build_search_model(dim=params.get("dim", 32),
                                  cheap_pooling=params.get("cheap_pooling", "none"),
                                  threshold=params.get("threshold", 0.5),
                                  uncertainty_on=True), True
    if arch.startswith("RUNE-04"):
        from training.models.adaptive import build_adaptive_model

        return build_adaptive_model(dim=params.get("dim", 32),
                                    cheap_pooling=params.get("cheap_pooling", "none"),
                                    threshold=params.get("threshold", 0.5)), False
    if arch.startswith("RUNE-03-"):
        from training.models.dense import build_dense_model

        return build_dense_model(variant=arch.split("-")[-1],
                                 token_dims=params.get("token_dims", [32] * 8),
                                 pooling=params.get("pooling", "none"),
                                 gate_on=params.get("gate_on", False),
                                 head_h1=params.get("head_h1", 128),
                                 head_h2=params.get("head_h2", 32)), False
    from training.models.rune_models import build_model

    return build_model(arch), False


def teacher_outputs(model, ids, masks, has_unc):
    out = model(ids, masks)
    if has_unc and len(out) == 7:
        _, _, rv, rw, _, u, _ = out
        return rv, rw, u
    if len(out) == 5:
        _, _, rv, rw, _ = out
        return rv, rw, None
    v, w = out
    return v, w, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--teacher-id", required=True)
    ap.add_argument("--params-json", default="{}")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--meta-out", default="")
    args = ap.parse_args()

    params = json.loads(args.params_json)
    model, has_unc = build_teacher_model(args.arch, params)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    with open(args.checkpoint, "rb") as f:
        ckpt_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    pool = P.load_jsonl(args.pool)
    loader, _ = make_loader(pool, batch_size=args.batch_size, shuffle=False, seed=0)
    labeled = []
    t0 = time.time()
    n = 0
    with torch.no_grad():
        for batch in loader:
            ids, masks, value, wdl = batch[:4]
            v, w, u = teacher_outputs(model, ids, masks, has_unc)
            wcls = w.argmax(dim=-1)
            for i in range(len(value)):
                r = dict(pool[n + i])
                r["teacher_v"] = float(v[i])
                r["teacher_w"] = int(wcls[i])
                if u is not None:
                    r["teacher_u"] = float(u[i])
                r["teacher_id"] = args.teacher_id
                labeled.append(r)
            n += len(value)
    dt = time.time() - t0
    P.save_jsonl(args.out, labeled)
    meta = {"teacher_id": args.teacher_id, "teacher_arch": args.arch,
            "teacher_ckpt": args.checkpoint, "teacher_hash": ckpt_hash,
            "positions": n, "seconds": dt,
            "positions_per_sec": n / max(1e-9, dt),
            "uncertainty_present": has_unc,
            "warning": "teacher supervision is not free; cost reported here, never amortized silently"}
    if args.meta_out:
        with open(args.meta_out, "w") as f:
            json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
