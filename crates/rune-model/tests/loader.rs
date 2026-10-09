use std::path::PathBuf;
#[test]
fn fixtures_have_format2_and_hash() {
    let names = [
        "tiny-mlp-fp32.rune",
        "small-gab-fp32.rune",
        "small-gab-int8.rune",
        "small-gab-int16.rune",
        "rel-08x32-fp32.rune",
        "dense-b-fp32.rune",
        "dense-b-int8.rune",
        "dense-b-int16.rune",
        "adaptive-fp32.rune",
    ];
    for n in names {
        let p = PathBuf::from(format!("../../spec/test-vectors/models/{}", n));
        let m = rune_model::load(&p).expect(n);
        assert_eq!(m.header.format, 2);
        assert_eq!(m.header.feature_version, "grouped_hkav2_fullthreats_v02");
        assert!(!m.header.model_hash.is_empty());
    }
}
#[test]
fn rejects_bad_magic() {
    let d = std::env::temp_dir().join("rune-bad-magic.rune");
    std::fs::write(&d, b"BAD!xxxx").unwrap();
    let r = rune_model::load(&d);
    assert!(r.is_err());
}
fn write_minimal(header_json: &str) -> PathBuf {
    let d = std::env::temp_dir().join("rune-game-gate.rune");
    let mut buf = Vec::from([b'R', b'U', b'N', b'E']);
    let hb = header_json.as_bytes();
    buf.extend_from_slice(&(hb.len() as u32).to_le_bytes());
    buf.extend_from_slice(hb);
    std::fs::write(&d, &buf).unwrap();
    d
}
#[test]
fn legacy_files_default_to_chess() {
    let p = PathBuf::from("../../spec/test-vectors/models/tiny-mlp-fp32.rune");
    let m = rune_model::load(&p).expect("fixture");
    assert_eq!(m.header.game, "chess");
}
#[test]
fn rejects_unknown_game() {
    let d = write_minimal(r#"{"format":2,"architecture_id":"RUNE-MLP","game":"unknown_game_xyz"}"#);
    let r = rune_model::load(&d);
    assert!(matches!(r, Err(rune_model::LoadError::GameMismatch(_))));
}
#[test]
fn known_non_chess_game_passes_gate() {
    let d = write_minimal(r#"{"format":2,"architecture_id":"RUNE-MLP","game":"shogi","feature_version":"shogi_raw_v01"}"#);
    let r = rune_model::load(&d);
    assert!(r.is_err());
    assert!(!matches!(r, Err(rune_model::LoadError::GameMismatch(_))));
}
#[test]
fn game_feature_versions() {
    assert_eq!(rune_spec::game_feature_version("chess"), Some("grouped_hkav2_fullthreats_v02"));
    assert_eq!(rune_spec::game_feature_version("shogi"), Some("shogi_raw_v01"));
    assert_eq!(rune_spec::game_feature_version("go"), Some("go_planes_v01"));
}
#[test]
fn rejects_truncated() {
    let src = PathBuf::from("../../spec/test-vectors/models/tiny-mlp-fp32.rune");
    let bytes = std::fs::read(&src).unwrap();
    let cut = bytes[..bytes.len() / 2].to_vec();
    let d = std::env::temp_dir().join("rune-trunc.rune");
    std::fs::write(&d, &cut).unwrap();
    let r = rune_model::load(&d);
    assert!(r.is_err());
}
#[test]
fn rejects_corrupt_hash() {
    let src = PathBuf::from("../../spec/test-vectors/models/tiny-mlp-fp32.rune");
    let mut bytes = std::fs::read(&src).unwrap();
    let n = bytes.len();
    bytes[n - 1] ^= 0xFF;
    let d = std::env::temp_dir().join("rune-corrupt.rune");
    std::fs::write(&d, &bytes).unwrap();
    let r = rune_model::load(&d);
    assert!(r.is_err());
}
