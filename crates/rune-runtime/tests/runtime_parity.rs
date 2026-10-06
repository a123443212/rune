use rune_runtime::board::Board;
use rune_runtime::evaluator::Evaluator;
use rune_runtime::features::{diff_features, extract_features};
use std::fs;
use std::path::PathBuf;
fn vec_file(name: &str) -> serde_json::Value {
    let p = format!("../../spec/test-vectors/v10/{}", name);
    let s = fs::read_to_string(&p).expect("read vector");
    serde_json::from_str(&s).expect("parse")
}
#[test]
fn features_match_golden() {
    let v = vec_file("features.json");
    for e in v["vectors"].as_array().unwrap() {
        let fen = e["fen"].as_str().unwrap();
        let b = Board::parse_fen(fen).expect("fen");
        let got = extract_features(&b);
        let want: Vec<(u8, u16)> = e["features"].as_array().unwrap().iter().map(|x| {
            (x[0].as_u64().unwrap() as u8, x[1].as_u64().unwrap() as u16)
        }).collect();
        assert_eq!(got, want, "{}", fen);
    }
}
#[test]
fn features_cover_special_moves() {
    let cases = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/P7/8/8/8/1k6/8/4K3 w - - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ];
    for fen in cases {
        let b = Board::parse_fen(fen).expect("fen");
        let f = extract_features(&b);
        assert!(!f.is_empty());
        let mut s = f.clone();
        s.sort_unstable();
        s.dedup();
        assert_eq!(f, s);
    }
}
#[test]
fn accumulator_refresh_equals_incremental() {
    let b0 = Board::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    let mut b1 = b0.clone();
    b1.apply_uci("e2e4").unwrap();
    let f0 = extract_features(&b0);
    let f1 = extract_features(&b1);
    let (added, removed) = diff_features(&f0, &f1);
    assert!(!added.is_empty() || !removed.is_empty());
    let p = PathBuf::from("../../spec/test-vectors/models/small-gab-fp32.rune");
    let mut ev_full = Evaluator::load(&p).expect("load");
    let mut ev_inc = Evaluator::load(&p).expect("load");
    ev_full.refresh(&b1);
    ev_inc.refresh(&b0);
    ev_inc.update_incremental(&f0, &f1);
    let r1 = ev_full.evaluate();
    let r2 = ev_inc.evaluate();
    assert!((r1.value - r2.value).abs() < 1e-6);
}
#[test]
fn make_unmake_roundtrip() {
    let mut b = Board::startpos();
    let k0 = b.to_fen();
    let moves = ["e2e4", "e7e5", "g1f3"];
    for m in moves {
        b.apply_uci(m).unwrap();
    }
    for _ in moves {
        b.unmake().unwrap();
    }
    assert_eq!(b.to_fen(), k0);
    let f0 = extract_features(&Board::startpos());
    let f1 = extract_features(&b);
    assert_eq!(f0, f1);
}
#[test]
fn model_fixtures_load_and_eval() {
    let names = [
        "tiny-mlp-fp32.rune",
        "small-gab-fp32.rune",
        "small-gab-int8.rune",
        "small-gab-int16.rune",
        "rel-08x32-fp32.rune",
    ];
    for n in names {
        let p = PathBuf::from(format!("../../spec/test-vectors/models/{}", n));
        let mut ev = Evaluator::load(&p).expect(n);
        let b = Board::startpos();
        let r = ev.evaluate_board(&b);
        assert!(r.value >= -1.0 && r.value <= 1.0, "{}", n);
    }
}
#[test]
fn startpos_value_matches_golden_triangle() {
    let p = PathBuf::from("../../spec/test-vectors/models/small-gab-fp32.rune");
    let mut ev = Evaluator::load(&p).expect("load");
    let b = Board::startpos();
    let r = ev.evaluate_board(&b);
    assert!((r.value + 0.019763).abs() < 2e-5, "got {}", r.value);
}
#[test]
fn quant_model_close_to_fp32() {
    let pf = PathBuf::from("../../spec/test-vectors/models/small-gab-fp32.rune");
    let pi = PathBuf::from("../../spec/test-vectors/models/small-gab-int8.rune");
    let mut ef = Evaluator::load(&pf).unwrap();
    let mut ei = Evaluator::load(&pi).unwrap();
    let b = Board::startpos();
    let rf = ef.evaluate_board(&b);
    let ri = ei.evaluate_board(&b);
    assert!((rf.value - ri.value).abs() < 0.01, "fp32 {} int8 {}", rf.value, ri.value);
}
#[test]
fn push_pop_roundtrip() {
    let p = PathBuf::from("../../spec/test-vectors/models/small-gab-fp32.rune");
    let mut ev = Evaluator::load(&p).expect("load");
    let b0 = Board::startpos();
    let mut b1 = b0.clone();
    b1.apply_uci("e2e4").unwrap();
    let f0 = extract_features(&b0);
    let f1 = extract_features(&b1);
    ev.refresh(&b0);
    let r0 = ev.evaluate();
    ev.push();
    ev.update_incremental(&f0, &f1);
    let r1 = ev.evaluate();
    ev.pop();
    assert_eq!(ev.search_depth(), 0);
    let r2 = ev.evaluate();
    assert!((r0.value - r2.value).abs() < 1e-6);
    assert!((r0.wdl[0] - r2.wdl[0]).abs() < 1e-6);
    let _ = r1;
}
#[test]
fn value_only_matches_full() {
    let p = PathBuf::from("../../spec/test-vectors/models/small-gab-fp32.rune");
    let mut ev = Evaluator::load(&p).expect("load");
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ] {
        let b = Board::parse_fen(fen).unwrap();
        ev.refresh(&b);
        let r = ev.evaluate();
        let v = ev.evaluate_value_only();
        assert!((r.value - v).abs() < 1e-6, "{} got {} vs {}", fen, r.value, v);
    }
}
#[test]
fn large_diff_matches_refresh() {
    let b0 = Board::startpos();
    let b1 = Board::parse_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
    let f0 = extract_features(&b0);
    let f1 = extract_features(&b1);
    let (added, removed) = diff_features(&f0, &f1);
    assert!(added.len() + removed.len() > 64);
    let p = PathBuf::from("../../spec/test-vectors/models/small-gab-fp32.rune");
    let mut ev_full = Evaluator::load(&p).expect("load");
    let mut ev_inc = Evaluator::load(&p).expect("load");
    ev_full.refresh(&b1);
    ev_inc.refresh(&b0);
    ev_inc.update_incremental(&f0, &f1);
    let r1 = ev_full.evaluate();
    let r2 = ev_inc.evaluate();
    assert!((r1.value - r2.value).abs() < 1e-5, "{} vs {}", r1.value, r2.value);
}
#[test]
fn unsupported_arch_fails_closed() {
    use rune_runtime::RuntimeError;
    for n in ["dense-b-fp32.rune", "dense-b-int8.rune", "dense-b-int16.rune", "adaptive-fp32.rune"] {
        let p = PathBuf::from(format!("../../spec/test-vectors/models/{}", n));
        let m = rune_model::load(&p).expect(n);
        let r = Evaluator::from_model(&m);
        assert!(matches!(r, Err(RuntimeError::UnsupportedArch(_))), "{}", n);
    }
}
