import json


def export_canonical_graph(ir, path):
    doc = {
        "ir_version": ir.get("ir_version"),
        "spec_version": ir.get("spec_version"),
        "model": ir.get("model"),
        "target": ir.get("target"),
        "ops": ir.get("ops"),
        "tensors": ir.get("tensors"),
        "fusion": ir.get("fusion", []),
        "kernel_plan": ir.get("kernel_plan", []),
        "memory": ir.get("memory", {}),
        "hashes": ir.get("hashes", {}),
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
    return path


def load_canonical_graph(path):
    with open(path, "r") as f:
        doc = json.load(f)
    return doc


def graph_summary(ir):
    model = ir.get("model", {})
    lines = []
    lines.append("arch %s tokens %d dim %d quant %s isa %s" % (
        model.get("architecture"),
        int(model.get("tokens", 0)),
        int(model.get("token_dim", 0)),
        model.get("quantization"),
        ir.get("target", {}).get("isa"),
    ))
    for op in ir.get("ops", []):
        lines.append("%s %s -> %s" % (op.get("id"), op.get("kind"), ",".join(op.get("outputs", []))))
    return "\n".join(lines)
