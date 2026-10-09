use super::version::{CLASSIC_REQUIRED_OPS, RESNET_REQUIRED_OPS};

pub fn arch_family(arch: &str) -> &'static str {
    if arch.starts_with("RUNE-RESNET") {
        "resnet"
    } else if arch == "RUNE-04" || arch == "RUNE-05" {
        "adaptive"
    } else if arch == "RUNE-SFNN"
        || arch == "RUNE-MLP"
        || arch == "RUNE-ATTN"
        || arch == "RUNE-ATTN-GAB"
        || arch == "RUNE-ATTN-MH4"
        || arch == "RUNE-REL-02"
        || arch == "RUNE-03"
        || arch.starts_with("RUNE-03-")
    {
        "classic"
    } else {
        "unknown"
    }
}

pub fn required_ops_for_arch(arch: &str) -> Vec<&'static str> {
    match arch_family(arch) {
        "resnet" => RESNET_REQUIRED_OPS.to_vec(),
        "classic" | "adaptive" => CLASSIC_REQUIRED_OPS.to_vec(),
        _ => Vec::new(),
    }
}

pub fn is_resnet_arch(arch: &str) -> bool {
    arch_family(arch) == "resnet"
}
