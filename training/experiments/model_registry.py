import warnings


MODEL_IDS = {
    "rune_mlp": "RUNE-MLP",
    "rune_attn": "RUNE-ATTN",
    "rune_attn_gab": "RUNE-ATTN-GAB",
    "rune_attn_gab_pair": "RUNE-ATTN-GAB",
    "rune_rel_s": "RUNE-REL-02",
    "rune_rel_d": "RUNE-REL-02",
    "rune_03A": "RUNE-03-A",
    "rune_03B": "RUNE-03-B",
    "rune_03C": "RUNE-03-C",
    "rune_03D": "RUNE-03-D",
    "rune_04_cheap": "RUNE-04",
    "rune_04_always": "RUNE-04",
    "rune_04_adaptive": "RUNE-04",
    "rune_05_unc": "RUNE-05",
    "rune_05_routing": "RUNE-05",
    "rune_05_stab": "RUNE-05",
    "rune_s1": "RUNE-03-A",
    "rune_s2": "RUNE-03-A",
    "rune_s3": "RUNE-03-A",
    "rune_s4": "RUNE-03-A",
}

CORE_MODELS = ("rune_mlp",)
EXPERIMENTAL_MODELS = (
    "rune_attn", "rune_attn_gab", "rune_rel_s", "rune_rel_d",
    "rune_03A", "rune_03B", "rune_03C", "rune_03D",
    "rune_04_cheap", "rune_04_always", "rune_04_adaptive",
    "rune_05_unc", "rune_05_routing", "rune_05_stab",
    "rune_s1", "rune_s2", "rune_s3", "rune_s4", "rune_attn_gab_pair",
)
PAIR_MODELS = frozenset(("rune_attn_gab_pair",))

ADAPTIVE_MODES = {
    "rune_04_cheap": "cheap",
    "rune_04_always": "always",
    "rune_04_adaptive": "adaptive",
    "rune_05_unc": "adaptive",
    "rune_05_routing": "adaptive",
    "rune_05_stab": "adaptive",
}

SEARCH_ROUTING = {
    "rune_05_unc": "difficulty",
    "rune_05_routing": "both",
    "rune_05_stab": "full",
}

CORE_ARCH = {"tokens": 8, "token_dim": 32, "gate": "clip", "alpha": 1.0}


def pair_enabled_for_model(config, model_key):
    for leg in config.get("pair_legs", {}).values():
        if leg.get("model") == model_key:
            return bool(leg.get("pair", False))
    if model_key in PAIR_MODELS:
        return True
    models = config.get("models", [])
    return len(models) == 1 and bool(config.get("architecture", {}).get("pair", False))


def experimental_reasons(config):
    reasons = []
    for model in config.get("models", []):
        if model in EXPERIMENTAL_MODELS:
            reasons.append(f"model {model} is experimental")
    loss = config.get("loss", {})
    if loss.get("ranking", False):
        reasons.append("ranking loss is experimental (use the L1 driver on candidates)")
    if loss.get("uncertainty", False):
        reasons.append("uncertainty loss is experimental (L1 leg only, calibration-gated)")
    if loss.get("stability", False):
        reasons.append("stability loss is experimental (L2 leg only, needs child data)")
    dist = config.get("distillation", {})
    if dist.get("enabled", False):
        reasons.append(f"distillation is experimental (task={dist.get('task', 'value_wdl')}, "
                       f"weighting={dist.get('weight_mode', 'uniform')})")
    sampling = config.get("sampling", {})
    if sampling.get("mode", "none") == "disagreement_mix":
        reasons.append("disagreement sampling is experimental (stage-gated after S0)")
    architecture = config.get("architecture", {})
    for key in ("tokens", "token_dim", "gate", "alpha"):
        if architecture.get(key, CORE_ARCH[key]) != CORE_ARCH[key]:
            warnings.warn(f"architecture.{key}={architecture.get(key)} deviates from core")
    return reasons