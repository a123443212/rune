import struct

import numpy as np

MAGIC = b"RUNE"
FORMAT_VERSION = 2
FEATURE_VERSION = "grouped_hkav2_fullthreats_v02"


def fnv1a(data):
    h = 1469598103934665603
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def symmetric_scale(arr, bits=8):
    m = float(abs(arr).max()) if arr.size else 0.0
    if m <= 0:
        return 1.0
    return m / (32767.0 if bits == 16 else 127.0)


def quantize_array(arr, bits=8):
    scale = symmetric_scale(arr, bits)
    bound = 32767 if bits == 16 else 127
    dtype = np.int16 if bits == 16 else np.int8
    flat = np.asarray(arr, dtype=np.float64) / scale
    pos = np.floor(flat + 0.5)
    neg = np.ceil(flat - 0.5)
    r = np.where(flat >= 0, pos, neg)
    q = np.clip(r, -bound, bound).astype(dtype)
    return q, scale


def collect_tensors(model, quantization="fp32"):
    from training.features.python_features import NUM_GROUPS

    arch_tensors = model.arch_tensors()
    emb_tensors = model.embedding_tensors()
    order = ["emb" + str(g) for g in range(NUM_GROUPS)] + model.export_order()
    arrays = {}
    for g in range(NUM_GROUPS):
        arrays[f"emb{g}"] = emb_tensors[f"emb{g}"].numpy()
    for k, v in arch_tensors.items():
        arrays[k] = np.asarray(v.numpy(), dtype=np.float32)
    return order, arrays


def export_model(model, path, quantization="fp32"):
    import json

    bits = 16 if quantization == "int16" else 8
    order, arrays = collect_tensors(model, quantization)
    spec = model.model_spec(quantization=quantization)
    tensors_meta = []
    scales = {}
    payload = bytearray()
    for name in order:
        arr = np.asarray(arrays[name], dtype=np.float32)
        if quantization in ("int8", "int16") and name.startswith("emb"):
            q, scale = quantize_array(arr, bits)
            scales[name] = scale
            tensors_meta.append({"name": name, "shape": list(q.shape),
                                 "dtype": "int16" if bits == 16 else "int8"})
            payload += q.tobytes()
        else:
            tensors_meta.append({"name": name, "shape": list(arr.shape), "dtype": "float32"})
            payload += arr.astype("<f4").tobytes()
    header = {
        "format": FORMAT_VERSION,
        "architecture_id": spec["arch"],
        "architecture_version": spec["arch_version"],
        "feature_version": spec.get("feature_set", FEATURE_VERSION),
        "arch": spec["arch"],
        "arch_version": spec["arch_version"],
        "feature_set": spec.get("feature_set", FEATURE_VERSION),
        "tokens": spec["tokens"],
        "token_dim": spec["token_dim"],
        "attention": spec["attention"],
        "geometric_bias": spec["geometric_bias"],
        "head": spec["head"],
        "quantization": quantization,
        "quantization_metadata": {"mode": "symmetric", "scales": scales},
        "scales": scales,
        "tensor_metadata": tensors_meta,
        "tensors": tensors_meta,
    }
    for key in ("gate", "alpha", "context_dim", "variant", "token_dims", "pooling",
                "pool_clip", "gate_on", "shared_width", "cheap_pooling",
                "threshold", "t_high", "t_low", "refine_precision",
                "pruned_pairs", "uncertainty", "stability_head", "head_h1",
                "head_h2", "cheap_hidden", "ref_h1", "ref_h2", "teacher_id",
                "teacher_hash", "student_of", "head_buckets"):
        if key in spec:
            header[key] = spec[key]
    hh = format(fnv1a(payload), "016x")
    header["model_hash"] = hh
    header["checksum"] = hh
    hbytes = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(hbytes)))
        f.write(hbytes)
        f.write(payload)
    return header


def read_header(path):
    import json

    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == MAGIC, "bad magic"
        (n,) = struct.unpack("<I", f.read(4))
        header = json.loads(f.read(n))
    return header


def load_exported_arrays(path):
    import json

    header = read_header(path)
    with open(path, "rb") as f:
        f.read(4)
        (n,) = struct.unpack("<I", f.read(4))
        f.read(n)
        payload = f.read()
    arrays = {}
    off = 0
    tlist = header.get("tensor_metadata", header.get("tensors"))
    for t in tlist:
        name, shape, dtype = t["name"], t["shape"], t["dtype"]
        count = 1
        for s in shape:
            count *= s
        if dtype == "int8":
            buf = payload[off:off + count]
            arr = np.frombuffer(buf, dtype=np.int8).reshape(shape)
            off += count
        elif dtype == "int16":
            buf = payload[off:off + count * 2]
            arr = np.frombuffer(buf, dtype="<i2").reshape(shape)
            off += count * 2
        else:
            buf = payload[off:off + count * 4]
            arr = np.frombuffer(buf, dtype="<f4").reshape(shape).astype(np.float32)
            off += count * 4
        arrays[name] = np.array(arr)
    return header, arrays
