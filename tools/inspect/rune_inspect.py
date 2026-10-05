import argparse
import json
import struct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    with open(args.model, "rb") as f:
        magic = f.read(4)
        assert magic == b"RUNE", "bad magic"
        (n,) = struct.unpack("<I", f.read(4))
        header = json.loads(f.read(n))
        payload = f.read()
    print("bytes %d payload %d" % (8 + n + len(payload), len(payload)))
    for k in ("architecture_id", "architecture_version", "tokens", "token_dim", "quantization", "compiled", "compiled_kind", "rune_ir_version", "compiler_version", "target_isa", "target_cpu", "source_hash", "kernel_plan_hash", "model_hash", "spec_version"):
        if k in header:
            print("%s %s" % (k, header[k]))
    plan = header.get("kernel_plan", [])
    print("kernels %d" % len(plan))
    for e in plan:
        print("kernel %s %s %s %s %s" % (e.get("kind"), e.get("kernel_id"), e.get("shape"), e.get("dtype"), e.get("isa")))
    mem = header.get("memory_plan", {})
    if mem:
        print("arena_bytes %s" % mem.get("arena_bytes"))
        print("strategy %s" % mem.get("strategy"))
    pre = header.get("precomputed", {})
    if pre:
        print("precomputed %s" % json.dumps(pre, sort_keys=True)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
