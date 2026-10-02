import numpy as np


def fake_quantize(arr, scale=None):
    arr = np.asarray(arr, dtype=np.float64)
    if scale is None:
        m = float(np.abs(arr).max()) if arr.size else 0.0
        scale = m / 127.0 if m > 0 else 1.0
    q = np.clip(np.round(arr / scale), -127, 127)
    return (q * scale).astype(np.float32), scale


def quantization_report(arrays):
    report = {}
    for name, arr in arrays.items():
        fq, scale = fake_quantize(np.asarray(arr, dtype=np.float32))
        err = np.abs(np.asarray(arr, dtype=np.float32) - fq)
        report[name] = {
            "scale": float(scale),
            "max_abs_err": float(err.max()) if err.size else 0.0,
            "mean_abs_err": float(err.mean()) if err.size else 0.0,
        }
    return report
