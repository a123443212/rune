pub const RUNTIME_SPEC: &str = "RUNE-10";
pub const MODEL_FORMAT_VERSION: u32 = 2;
pub const FEATURE_VERSION: &str = "grouped_hkav2_fullthreats_v02";
pub const GAME_CHESS: &str = "chess";
pub const GAME_SHOGI: &str = "shogi";
pub const GAME_XIANGQI: &str = "xiangqi";
pub fn game_feature_version(game: &str) -> Option<&'static str> {
    match game {
        "chess" => Some(FEATURE_VERSION),
        "shogi" => Some("shogi_raw_v01"),
        "xiangqi" => Some("xiangqi_raw_v01"),
        _ => None,
    }
}
pub const CONTEXT_DIM: usize = 17;
pub const NUM_GROUPS: usize = 9;
pub const TOKEN_DIM_DEFAULT: usize = 32;
pub const VOCAB_SIZES: [usize; 9] = [256, 256, 256, 128, 128, 512, 512, 64, 4560];
pub const GROUP_NAMES: [&str; 9] = [
    "pawn_structure",
    "king_zone",
    "minor_pieces",
    "rooks",
    "queens",
    "threats",
    "mobility",
    "global",
    "pawn_pairs",
];
pub const MAGIC: [u8; 4] = [82, 85, 78, 69];
pub const MAX_HEADER_LEN: usize = 1000000;
pub const MAX_TENSOR_ELEMS: usize = 200000000;
pub const MAX_PAYLOAD_BYTES: usize = 800000000;
pub fn vocab_size(group: usize) -> usize {
    VOCAB_SIZES[group]
}
pub fn is_supported_format(v: u32) -> bool {
    v == 1 || v == 2
}
pub fn is_supported_quant(q: &str) -> bool {
    q == "fp32" || q == "int8" || q == "int16"
}
pub fn quant_bound(quant: &str) -> i32 {
    if quant == "int16" {
        32767
    } else {
        127
    }
}
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for b in data {
        h ^= *b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}
pub fn fnv1a_str(s: &str) -> u64 {
    fnv1a64(s.as_bytes())
}
pub mod tolerance {
    pub const MATVEC_ABS: f32 = 1e-5;
    pub const HEAD_ABS: f32 = 2e-5;
    pub const TANH_ABS: f32 = 2e-6;
    pub const RESIDUAL_ABS: f32 = 1e-6;
    pub const GATE_FMA_ABS: f32 = 1e-6;
}
