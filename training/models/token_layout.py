from training.features.python_features import VOCAB_SIZES

TOKEN_COUNTS = (6, 8, 10)
TOKEN_DIMS = (24, 32, 40)


def _full(group):
    return [(group, 0, VOCAB_SIZES[group])]


def _split(group, lo, hi):
    return [(group, lo, hi)]


def layout_for(tokens):
    if tokens == 8:
        return [_full(g) for g in range(8)]
    if tokens == 6:
        return [_full(0), _full(1), _full(2), _full(3) + _full(4),
                _full(5) + _full(6), _full(7)]
    if tokens == 10:
        return [_full(0), _full(1), _split(2, 0, 128), _split(2, 128, 256),
                _full(3), _full(4), _split(5, 0, 384), _split(5, 384, 512),
                _full(6), _full(7)]
    raise ValueError(f"unsupported token count {tokens}")


def check_layout(tokens, dim):
    if tokens not in TOKEN_COUNTS:
        raise ValueError(f"unsupported token count {tokens}")
    if dim not in TOKEN_DIMS:
        raise ValueError(f"unsupported token dim {dim}")
    return layout_for(tokens)
