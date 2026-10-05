import math

MATE_GUARD = 9000
SCORE_SCALE = 1000.0
VALUE_MIN = -1.0
VALUE_MAX = 1.0


def canonical_value(raw):
    x = float(raw)
    if math.isnan(x):
        return 0.0
    if x < VALUE_MIN:
        return VALUE_MIN
    if x > VALUE_MAX:
        return VALUE_MAX
    return x


def engine_score(value):
    v = canonical_value(value)
    m = math.trunc(abs(v) * SCORE_SCALE + 0.5)
    s = int(m) if v >= 0 else -int(m)
    if s >= MATE_GUARD:
        s = MATE_GUARD - 1
    if s <= -MATE_GUARD:
        s = -MATE_GUARD + 1
    return s


def wdl_normalize(wdl):
    w = [float(wdl[0]), float(wdl[1]), float(wdl[2])]
    s = w[0] + w[1] + w[2]
    if s <= 0.0:
        return [0.0, 1.0, 0.0]
    return [w[0] / s, w[1] / s, w[2] / s]


def wdl_from_value(value):
    v = canonical_value(value)
    w = (v + 1.0) * 0.5
    return [w * w, 1.0 - w * w - (1.0 - w) * (1.0 - w), (1.0 - w) * (1.0 - w)]


def value_from_wdl(wdl):
    w = wdl_normalize(wdl)
    return w[0] - w[2]


def value_wdl_consistent(value, wdl, tol=0.35):
    return abs(canonical_value(value) - value_from_wdl(wdl)) <= tol


def is_extreme(value, bound=0.95):
    return abs(canonical_value(value)) >= bound
