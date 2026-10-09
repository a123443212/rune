IR_VERSION = "1.1"
SPEC_VERSION = "RUNE-11"
COMPILER_VERSION = "0.12.0"

CLASSIC_REQUIRED = ["FeatureUpdate", "AccumulatorUpdate", "Tokenize", "Q", "K", "V", "Score", "Bias", "Gate", "Mix", "Residual", "HeadH1", "HeadH2", "Value", "WDL"]
RESNET_REQUIRED = ["StemConv", "Conv2D", "ResidualAdd", "Relu", "Value", "WDL", "PolicyLogits", "Policy"]

VALID_ISA = ["portable", "avx2", "avx512"]


def arch_family(arch):
    if arch.startswith("RUNE-RESNET"):
        return "resnet"
    if arch in ("RUNE-04", "RUNE-05"):
        return "adaptive"
    if arch in ("RUNE-SFNN", "RUNE-MLP", "RUNE-ATTN", "RUNE-ATTN-GAB", "RUNE-ATTN-MH4", "RUNE-REL-02", "RUNE-03", "RUNE-04", "RUNE-05") or arch.startswith("RUNE-03-"):
        return "classic"
    return "unknown"


def required_ops(arch):
    fam = arch_family(arch)
    if fam == "resnet":
        return list(RESNET_REQUIRED)
    if fam in ("classic", "adaptive"):
        return list(CLASSIC_REQUIRED)
    return []
