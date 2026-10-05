FUSED_GROUPS = [
    {"id": "f_qkv", "ops": ["Q", "K", "V"], "kernel": "qkv_fused_8x32"},
    {"id": "f_score_bias_gate", "ops": ["Score", "Bias", "Gate"], "kernel": "score_bias_gate_8x8"},
    {"id": "f_mix_residual", "ops": ["Mix", "Residual"], "kernel": "mix_residual_8x32"},
    {"id": "f_head_h1", "ops": ["HeadH1"], "kernel": "linear_bias_clip"},
    {"id": "f_head_h2", "ops": ["HeadH2"], "kernel": "linear_bias_clip"},
]


def _shape_key(model, kind):
    tokens = int(model.get("tokens", 8))
    dim = int(model.get("token_dim", 32))
    h1 = int(model.get("head_h1", 128))
    h2 = int(model.get("head_h2", 32))
    if kind in ("Q", "K", "V"):
        return "%dx%d" % (tokens, dim)
    if kind in ("Score", "Bias", "Gate"):
        return "%dx%d" % (tokens, tokens)
    if kind in ("Mix", "Residual"):
        return "%dx%d" % (tokens, dim)
    if kind == "HeadH1":
        return "%dx%d" % (h1, tokens * dim)
    if kind == "HeadH2":
        return "%dx%d" % (h2, h1)
    if kind == "Value":
        return "1x%d" % h2
    if kind == "WDL":
        return "3x%d" % h2
    return "%dx%d" % (tokens, dim)


def _kernel_for(kind, model, isa, quant):
    sk = _shape_key(model, kind)
    base = {
        "Q": "qkv_fused_8x32" if sk == "8x32" else "matvec_generic",
        "K": "qkv_fused_8x32" if sk == "8x32" else "matvec_generic",
        "V": "qkv_fused_8x32" if sk == "8x32" else "matvec_generic",
        "Score": "score_bias_gate_8x8" if sk == "8x8" else "matmul_tt_generic",
        "Bias": "score_bias_gate_8x8" if sk == "8x8" else "bias_add_generic",
        "Gate": "score_bias_gate_8x8" if sk == "8x8" else "gate_generic",
        "Mix": "mix_residual_8x32" if sk == "8x32" else "matmul_generic",
        "Residual": "mix_residual_8x32" if sk == "8x32" else "residual_generic",
        "HeadH1": "linear_bias_clip",
        "HeadH2": "linear_bias_clip",
        "Value": "dot_tanh",
        "WDL": "matvec_generic",
        "FeatureUpdate": "feature_pack",
        "AccumulatorUpdate": "accum_grouped",
        "Tokenize": "dequant_clip" if quant in ("int8", "int16") else "clip",
        "Route": "route_compare",
        "Cheap": "linear_bias_clip",
        "Refine": "refine_path",
    }
    kid = base.get(kind, "generic")
    return kid, sk


def select_kernels(ir):
    model = ir.get("model", {})
    isa = ir.get("target", {}).get("isa", "portable")
    quant = model.get("quantization", "fp32")
    plan = []
    fusion = []
    fused = {"Q": "f_qkv", "K": "f_qkv", "V": "f_qkv", "Score": "f_score_bias_gate", "Bias": "f_score_bias_gate", "Gate": "f_score_bias_gate", "Mix": "f_mix_residual", "Residual": "f_mix_residual"}
    seen_fusion = set()
    for op in ir.get("ops", []):
        kind = op.get("kind")
        kid, sk = _kernel_for(kind, model, isa, quant)
        packing = "row-major-aligned32"
        if kind in ("Q", "K", "V", "HeadH1", "HeadH2"):
            packing = "row-major-aligned32"
        entry = {
            "op": op.get("id"),
            "kind": kind,
            "kernel_id": kid,
            "shape": sk,
            "dtype": model.get("dtype", "fp32"),
            "packing": packing,
            "isa": isa,
            "fusion_group": fused.get(kind, ""),
        }
        plan.append(entry)
        fg = fused.get(kind, "")
        if fg and fg not in seen_fusion:
            seen_fusion.add(fg)
            for g in FUSED_GROUPS:
                if g["id"] == fg:
                    fusion.append(g)
    return plan, fusion


def cost_model(entry, isa):
    shape = entry.get("shape", "")
    kid = entry.get("kernel_id", "")
    base_ops = 1.0
    try:
        parts = shape.replace("x", " ").split()
        if len(parts) == 2:
            base_ops = float(int(parts[0]) * int(parts[1]))
    except Exception:
        base_ops = 1.0
    mem = base_ops * 4.0
    if "fused" in kid or kid.startswith("score") or kid.startswith("mix") or kid.startswith("qkv"):
        mem = mem * 0.6
    isa_factor = 1.0
    if isa == "avx2":
        isa_factor = 0.35
    elif isa == "avx512":
        isa_factor = 0.28
    return {"flops": base_ops, "bytes": mem, "isa_factor": isa_factor, "score": base_ops * isa_factor + mem * 0.05}
