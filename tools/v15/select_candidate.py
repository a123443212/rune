import sys

sys.path.insert(0, ".")

MACHINE = "linux-x86_64-cmake-O2-scalar"
DATE = "2026-10-05"

ARCH_PARAMS = {
    "C0-MLP": 37156,
    "C1-ATTN-GAB-8x32": 40388,
    "C2-Adaptive-04": 49001,
    "C3-Dense-B": 42592,
}

CANDIDATES = {
    "C0-MLP": {
        "params": None,
        "flops_k": 74,
        "full_us": 45.0,
        "search": "530n/255e d2, 30k nps",
        "evidence": "baseline, all langs, fixtures",
    },
    "C1-ATTN-GAB-8x32": {
        "params": None,
        "flops_k": 132,
        "full_us": 58.3,
        "incr_us": 44.9,
        "search": "266n/123e d2, 27.5k nps, det 3/3",
        "evidence": "triangle 9/9, simd 3.7x, fixtures fp32/int8/int16",
    },
    "C2-Adaptive-04": {
        "params": 49001,
        "flops_k": 98,
        "full_us": 57.8,
        "cheap_us": 7.9,
        "search": "not wired (void)cfg",
        "evidence": "routing 100%, refine-rate unmeasured trained",
    },
    "C3-Dense-B": {
        "params": 42592,
        "flops_k": 84,
        "full_us": None,
        "search": "rejected by search tool",
        "evidence": "int8 parity 7e-6, no search bench",
    },
}


def main():
    rows = []
    for name, c in CANDIDATES.items():
        c["params"] = ARCH_PARAMS[name]
    print(f"machine={MACHINE} date={DATE}")
    print("candidate,params,flops_k,full_us,notes")
    for name, c in CANDIDATES.items():
        print(f"{name},{c['params']},{c['flops_k']},{c['full_us']},{c['evidence']}")
    print("quality: unmeasured for all (no trained weights >=25M exist)")
    print("selection: C1-ATTN-GAB-8x32")


if __name__ == "__main__":
    main()
