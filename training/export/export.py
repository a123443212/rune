import struct

import numpy as np

MAGIC = b"RUNE"
FORMAT_VERSION = 1


def symmetric_scale(arr, bits=8):
    m = float(abs(arr).max()) if arr.size else 0.0
    if m <= 0:
        return 1.0
    return m / (32767.0 if bits == 16 else 127.0)


def quantize_array(arr, bits=8):
    scale = symmetric_scale(arr, bits)
    bound = 32767 if bits == 16 else 127
    dtype = np.int16 if bits == 16 else np.int8
    q = np.clip(np.round(arr / scale), -bound, bound).astype(dtype)
    return q, scale


def collect_tensors(model, quantization="fp32"):
    arch_tensors = model.arch_tensors()
    emb_tensors = model.embedding_tensors()
    order = ["emb" + str(g) for g in range(8)] + model.export_order()
    arrays = {}
    for g in range(8):
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
        "arch": spec["arch"],
        "arch_version": spec["arch_version"],
        "feature_set": spec["feature_set"],
        "tokens": spec["tokens"],
        "token_dim": spec["token_dim"],
        "attention": spec["attention"],
        "geometric_bias": spec["geometric_bias"],
        "head": spec["head"],
        "quantization": quantization,
        "scales": scales,
        "tensors": tensors_meta,
    }
    for key in ("gate", "alpha", "context_dim"):
        if key in spec:
            header[key] = spec[key]
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
    for t in header["tensors"]:
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
