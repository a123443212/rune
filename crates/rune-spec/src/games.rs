pub const RUNTIME_SPEC: &str = "RUNE-11";
pub const FEATURE_VERSION: &str = "grouped_hkav2_fullthreats_v02";
pub const GAME_CHESS: &str = "chess";
pub const GAME_SHOGI: &str = "shogi";
pub const GAME_XIANGQI: &str = "xiangqi";
pub const GAME_GO: &str = "go";

pub fn game_feature_version(game: &str) -> Option<&'static str> {
    match game {
        "chess" => Some(FEATURE_VERSION),
        "shogi" => Some("shogi_raw_v01"),
        "xiangqi" => Some("xiangqi_raw_v01"),
        "go" => Some("go_planes_v01"),
        _ => None,
    }
}

pub const CONTEXT_DIM: usize = 17;
pub const NUM_GROUPS: usize = 9;
pub const TOKEN_DIM_DEFAULT: usize = 32;
pub const VOCAB_SIZES: [usize; 9] = [256, 256, 256, 128, 128, 512, 512, 64, 4560];
pub const GROUP_NAMES: [&str; 9] = ["pawn_structure", "king_zone", "minor_pieces", "rooks", "queens", "threats", "mobility", "global", "pawn_pairs"];
