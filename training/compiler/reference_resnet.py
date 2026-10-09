import numpy as np


def relu(x):
    return np.maximum(x, 0.0)


def conv2d_nchw(x, w, b, pad=1):
    n, cin, h, wd = x.shape
    cout, _, kh, kw = w.shape
    oh, ow = h, wd
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((n, cout, oh, ow), dtype=np.float32)
    for oy in range(oh):
        for ox in range(ow):
            patch = xp[:, :, oy:oy + kh, ox:ox + kw]
            out[:, :, oy, ox] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3])) + b
    return out


def softmax(logits):
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    s = e.sum(axis=-1, keepdims=True)
    s[s <= 0] = 1.0
    return e / s


def forward_resnet(arrays, board, channels, blocks, policy_size, planes):
    import math
    x = relu(conv2d_nchw(planes, arrays["stem_w"], arrays["stem_b"]))
    for i in range(blocks):
        h = relu(conv2d_nchw(x, arrays[f"b{i}_w1"], arrays[f"b{i}_b1"]))
        h = conv2d_nchw(h, arrays[f"b{i}_w2"], arrays[f"b{i}_b2"])
        x = relu(x + h)
    pooled = x.mean(axis=(2, 3))
    flat = x.reshape(x.shape[0], -1)
    vh1 = np.asarray(arrays["vh1"]).reshape(-1, channels)
    bh1 = np.asarray(arrays["bh1"]).reshape(-1)
    h = np.minimum(np.maximum(pooled[0].dot(vh1.T) + bh1, 0.0), 1.0)
    wv = np.asarray(arrays["wv"]).reshape(-1)
    bv = float(np.asarray(arrays["bv"]).reshape(-1)[0])
    value = math.tanh(float((h * wv[:h.shape[0]]).sum() + bv))
    wwdl = np.asarray(arrays["wwdl"]).reshape(3, -1)
    bwdl = np.asarray(arrays["bwdl"]).reshape(-1)
    wdl = wwdl.dot(h[:wwdl.shape[1]]) + bwdl
    logits = np.asarray(arrays["wpol"]).reshape(policy_size, -1).dot(flat[0]) + np.asarray(arrays["bpol"]).reshape(-1)
    probs = softmax(logits.reshape(1, -1))[0]
    return {"value": value, "wdl": wdl, "logits": logits, "policy": probs}
