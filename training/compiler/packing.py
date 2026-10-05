import numpy as np

PACKING_VERSION = 1

PACKED_LAYOUTS = {
    "wq": "row-major-aligned32",
    "wk": "row-major-aligned32",
    "wvv": "row-major-aligned32",
    "w1": "row-major-aligned32",
    "w2": "row-major-aligned32",
}


def aligned_size(n, align=8):
    return ((n + align - 1) // align) * align


def pack_matrix_row_major(arr, align=8):
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        n = a.shape[0]
        pad = aligned_size(n) - n
        if pad:
            a = np.concatenate([a, np.zeros((pad,), dtype=np.float32)])
        return a
    rows, cols = a.shape
    pc = aligned_size(cols, align)
    if pc == cols:
        return np.ascontiguousarray(a, dtype=np.float32)
    out = np.zeros((rows, pc), dtype=np.float32)
    out[:, :cols] = a
    return np.ascontiguousarray(out)


def transpose_packed(arr):
    a = np.asarray(arr, dtype=np.float32)
    return np.ascontiguousarray(a.T)


def pack_weights(arrays, tensor_metas, isa):
    packed = {}
    layouts = {}
    for tm in tensor_metas:
        name = tm.get("name")
        if name not in arrays:
            continue
        arr = np.asarray(arrays[name], dtype=np.float32)
        if name in ("wq", "wk", "wvv", "w1", "w2", "wwdl"):
            if arr.ndim == 2:
                packed[name] = pack_matrix_row_major(arr, 8)
                layouts[name] = PACKED_LAYOUTS.get(name, "row-major-aligned32")
            else:
                packed[name] = np.ascontiguousarray(arr, dtype=np.float32)
                layouts[name] = "row-major-aligned32"
        elif name in ("bq", "bk", "bvv", "bv", "b1", "b2", "bvo", "bwdl", "wvo"):
            packed[name] = np.ascontiguousarray(arr, dtype=np.float32)
            layouts[name] = "row-major-aligned32"
        elif name in ("gabS", "gab"):
            packed[name] = np.ascontiguousarray(arr, dtype=np.float32)
            layouts[name] = "row-major-aligned32"
        elif name.startswith("emb"):
            packed[name] = np.ascontiguousarray(arr, dtype=np.float32)
            layouts[name] = "row-major-aligned32"
        else:
            packed[name] = np.ascontiguousarray(arr, dtype=np.float32)
            layouts[name] = "row-major-aligned32"
    meta = {
        "packing_version": PACKING_VERSION,
        "isa": isa,
        "layouts": layouts,
    }
    return packed, meta
