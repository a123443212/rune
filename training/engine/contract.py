import math


class EvalOutput:
    def __init__(self, value, wdl, uncertainty=0.0, refine=False, mode="full"):
        self.value = float(value)
        self.wdl = [float(wdl[0]), float(wdl[1]), float(wdl[2])]
        self.uncertainty = float(uncertainty)
        self.refine = bool(refine)
        self.mode = str(mode)

    def to_minimal(self):
        return (self.value, self.wdl)

    def to_dict(self):
        return {"value": self.value, "wdl": list(self.wdl), "uncertainty": self.uncertainty, "refine": self.refine, "mode": self.mode}


def make_output(value, wdl, uncertainty=0.0, refine=False, mode="full"):
    v = max(-1.0, min(1.0, float(value)))
    w = [float(wdl[0]), float(wdl[1]), float(wdl[2])]
    s = w[0] + w[1] + w[2]
    if s != 0.0:
        w = [x / s for x in w]
    return EvalOutput(v, w, uncertainty, refine, mode)


def needs_only(out, keys):
    d = out.to_dict()
    return {k: d[k] for k in keys if k in d}


def is_debug_key(key):
    return key in ("trace", "tokens", "mixer", "head", "features")
