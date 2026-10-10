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

import hashlib
import json
import os
import struct
import time

import numpy as np

from training.compiler.ir import COMPILER_VERSION, IR_VERSION

MAGIC = b"RUNE"
COMPILED_KIND = "compiled-v11"


def fnv1a64(data):
    h = 1469598103934665603
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def source_hash(header, payload):
    hb = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    h = hashlib.sha256()
    h.update(hb)
    h.update(bytes(payload))
    return h.hexdigest()[:16]


def plan_hash(kernel_plan, memory_plan, target):
    doc = json.dumps({"k": kernel_plan, "m": memory_plan, "t": target}, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(doc.encode())
    return h.hexdigest()[:16]


def cache_key(src_hash, arch, quant, isa, compiler_version=COMPILER_VERSION):
    raw = "%s|%s|%s|%s|%s" % (src_hash, arch, quant, isa, compiler_version)
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return h


def _dtype_to_payload(arr, dtype):
    if dtype == "int8":
        return np.asarray(arr, dtype=np.int8).tobytes()
    if dtype == "int16":
        return np.asarray(arr, dtype="<i2").tobytes()
    return np.asarray(arr, dtype="<f4").tobytes()


def write_compiled(path, rune_header, packed_arrays, order, ir, precomputed):
    header = dict(rune_header)
    kernel_plan = ir.get("kernel_plan", [])
    memory_plan = ir.get("memory", {})
    target = ir.get("target", {})
    payload = bytearray()
    tensors_meta = []
    for name in order:
        arr = packed_arrays.get(name)
        if arr is None:
            continue
        a = np.ascontiguousarray(arr)
        if a.dtype == np.int8:
            dtype = "int8"
        elif a.dtype == np.int16:
            dtype = "int16"
        else:
            dtype = "float32"
            a = np.asarray(a, dtype=np.float32)
        shape = list(a.shape)
        tensors_meta.append({"name": name, "shape": shape, "dtype": dtype})
        if dtype == "int8":
            payload += a.astype(np.int8).tobytes()
        elif dtype == "int16":
            payload += a.astype("<i2").tobytes()
        else:
            payload += a.astype("<f4").tobytes()
    src = source_hash({k: rune_header.get(k) for k in ("architecture_id", "model_hash", "tokens", "token_dim") if k in rune_header}, bytes(payload))
    kh = plan_hash(kernel_plan, memory_plan, target)
    header["tensor_metadata"] = tensors_meta
    header["tensors"] = tensors_meta
    header["compiled"] = True
    header["compiled_kind"] = COMPILED_KIND
    header["rune_ir_version"] = ir.get("ir_version", IR_VERSION)
    header["compiler_version"] = COMPILER_VERSION
    header["target_isa"] = target.get("isa", "portable")
    header["target_cpu"] = target.get("cpu", "generic-x86-64")
    header["kernel_plan"] = kernel_plan
    header["fusion_plan"] = ir.get("fusion", [])
    header["memory_plan"] = memory_plan
    header["packing_meta"] = precomputed.get("packing_meta", {})
    header["precomputed"] = precomputed.get("constants", {})
    header["source_hash"] = src
    header["kernel_plan_hash"] = kh
    header["spec_version"] = ir.get("spec_version", "RUNE-10")
    header["compile_time_utc"] = int(time.time())
    hh = format(fnv1a64(bytes(payload)), "016x")
    header["model_hash"] = hh
    header["checksum"] = hh
    hbytes = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(hbytes)))
        f.write(hbytes)
        f.write(bytes(payload))
    ck = cache_key(src, str(header.get("architecture_id")), str(header.get("quantization")), str(header.get("target_isa")))
    return {"path": path, "source_hash": src, "plan_hash": kh, "cache_key": ck, "bytes": 8 + len(hbytes) + len(payload)}


def read_compiled_header(path):
    import json as _json
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == MAGIC
        (n,) = struct.unpack("<I", f.read(4))
        header = _json.loads(f.read(n))
    return header


def cache_path(cache_dir, key):
    return os.path.join(cache_dir, "rune-%s.json" % key)
