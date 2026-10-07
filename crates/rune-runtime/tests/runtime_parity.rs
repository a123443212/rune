use rune_runtime::board::Board;
use rune_runtime::evaluator::Evaluator;
use rune_runtime::features::{compute_context, diff_features, extract_features};
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
    ev_inc.update_incremental(&f0, &f1, &compute_context(&b1));
    let r1 = ev_full.evaluate();
    let r2 = ev_inc.evaluate();
    assert!((r1.value - r2.value).abs() < 1e-6);
}
#[test]
fn clocks_roundtrip_and_update() {
    let mut b = Board::parse_fen("6k1/8/8/8/8/8/8/K6R w - - 100 45").unwrap();
    assert_eq!(b.halfmove_clock, 100);
    assert_eq!(b.fullmove_number, 45);
    assert_eq!(b.to_fen(), "6k1/8/8/8/8/8/8/K6R w - - 100 45");
    b.apply_uci("h1h2").unwrap();
    assert_eq!(b.halfmove_clock, 101);
    assert_eq!(b.fullmove_number, 45);
    b.unmake().unwrap();
    assert_eq!(b.to_fen(), "6k1/8/8/8/8/8/8/K6R w - - 100 45");
    let mut c = Board::startpos();
    c.apply_uci("e2e4").unwrap();
    assert_eq!(c.halfmove_clock, 0);
    assert_eq!(c.fullmove_number, 1);
    c.apply_uci("e7e5").unwrap();
    assert_eq!(c.fullmove_number, 2);
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
    ev.update_incremental(&f0, &f1, &compute_context(&b1));
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
    ev_inc.update_incremental(&f0, &f1, &compute_context(&b1));
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
#[test]
fn context_matches_reference_startpos() {
    let b = Board::startpos();
    let ctx = compute_context(&b);
    let want = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.625, 1.0];
    assert_eq!(ctx.len(), 8);
    for i in 0..8 {
        assert!((ctx[i] - want[i]).abs() < 1e-6, "{} got {} want {}", i, ctx[i], want[i]);
    }
}
#[test]
fn token_layout_matches_cpp_rules() {
    use rune_runtime::accumulator::token_of;
    for g in 0..8 {
        assert_eq!(token_of(8, g, 0), Some(g as usize));
    }
    assert_eq!(token_of(6, 0, 10), Some(0));
    assert_eq!(token_of(6, 2, 200), Some(2));
    assert_eq!(token_of(6, 3, 0), Some(3));
    assert_eq!(token_of(6, 4, 127), Some(3));
    assert_eq!(token_of(6, 5, 0), Some(4));
    assert_eq!(token_of(6, 6, 511), Some(4));
    assert_eq!(token_of(6, 7, 63), Some(5));
    assert_eq!(token_of(10, 0, 5), Some(0));
    assert_eq!(token_of(10, 1, 5), Some(1));
    assert_eq!(token_of(10, 2, 0), Some(2));
    assert_eq!(token_of(10, 2, 127), Some(2));
    assert_eq!(token_of(10, 2, 128), Some(3));
    assert_eq!(token_of(10, 2, 255), Some(3));
    assert_eq!(token_of(10, 3, 9), Some(4));
    assert_eq!(token_of(10, 4, 9), Some(5));
    assert_eq!(token_of(10, 5, 0), Some(6));
    assert_eq!(token_of(10, 5, 383), Some(6));
    assert_eq!(token_of(10, 5, 384), Some(7));
    assert_eq!(token_of(10, 5, 511), Some(7));
    assert_eq!(token_of(10, 6, 9), Some(8));
    assert_eq!(token_of(10, 7, 9), Some(9));
    assert_eq!(token_of(7, 0, 0), None);
    assert_eq!(token_of(6, 9, 0), None);
}
#[test]
fn six_token_accumulator_merges_groups() {
    use rune_runtime::accumulator::{Accumulator, Tables};
    use rune_spec::VOCAB_SIZES;
    let mut tables = Tables::zeros(4, VOCAB_SIZES);
    tables.data[3][0] = 1.0;
    tables.data[4][4] = 2.0;
    tables.data[4][5] = 4.0;
    tables.data[5][0] = 8.0;
    let mut acc = Accumulator::new(6, 4);
    acc.refresh(&tables, &[(3, 0), (4, 1), (5, 0)]);
    let raw = acc.raw();
    assert_eq!(raw.len(), 24);
    assert!((raw[12] - 3.0).abs() < 1e-6);
    assert!((raw[13] - 4.0).abs() < 1e-6);
    assert!((raw[16] - 8.0).abs() < 1e-6);
}
#[test]
fn dynamic_bias_matches_cache() {
    use rune_runtime::relational_cache::{IncrWeights, RelationalCache};
    let p = PathBuf::from("../../spec/test-vectors/models/rel-08x32-fp32.rune");
    let mut m = rune_model::load(&p).expect("load");
    let t = m.header.tokens;
    let d = m.header.token_dim;
    let cd = 8;
    m.arrays.insert("dynU".to_string(), vec![0.02; t * cd]);
    m.arrays.insert("dynW".to_string(), vec![0.03; t * cd]);
    m.header.raw["context_dim"] = serde_json::json!(8);
    let mut ev = Evaluator::from_model(&m).expect("dyn model loads");
    let gate_hard = m.header.raw.get("gate").and_then(|x| x.as_str()).unwrap_or("clip") == "hard_sigmoid";
    let alpha = m.header.raw.get("alpha").and_then(|x| x.as_f64()).unwrap_or(1.0) as f32;
    let pick = |base: &str, alt: &str| m.arrays.get(base).or_else(|| m.arrays.get(alt)).unwrap().clone();
    let gab = m.arrays.get("gabS").or_else(|| m.arrays.get("gab")).unwrap().clone();
    let w = IncrWeights {
        tokens: t,
        dim: d,
        wq: pick("wq", "wq"),
        bq: pick("bq", "bq"),
        wk: pick("wk", "wk"),
        bk: pick("bk", "bk"),
        wv: pick("wvv", "wv"),
        bv: pick("bvv", "bv"),
        gab,
        dyn_u: m.arrays.get("dynU").unwrap().clone(),
        dyn_w: m.arrays.get("dynW").unwrap().clone(),
        ctx_dim: cd,
        gate_hard,
        alpha,
    };
    let b = Board::parse_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
    ev.refresh(&b);
    let tr = ev.trace();
    let ctx = compute_context(&b);
    let mut cache = RelationalCache::configure(w, 8);
    cache.rebuild(&tr.tokens, Some(&ctx));
    let out = cache.out().to_vec();
    assert_eq!(out.len(), tr.mixer.mixed.len());
    let mut worst = 0.0;
    for i in 0..out.len() {
        let dd = (out[i] - tr.mixer.mixed[i]).abs();
        if dd > worst {
            worst = dd;
        }
    }
    assert!(worst < 1e-5, "worst {}", worst);
    let r = ev.evaluate();
    assert!((r.value - tr.value).abs() < 1e-7);
}
