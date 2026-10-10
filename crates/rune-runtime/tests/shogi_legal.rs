const START: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

#[test]
fn startpos_counts() {
    let black = rune_runtime::shogi::legal::legal_moves(START).expect("gen");
    assert_eq!(black.len(), 30);
    let mut sorted = black.clone();
    sorted.sort();
    assert_eq!(black, sorted);
    let nxt = rune_runtime::shogi::legal::apply_move(START, &black[0]).expect("apply");
    let white = rune_runtime::shogi::legal::legal_moves(&nxt).expect("gen");
    assert_eq!(white.len(), 30);
}

#[test]
fn apply_checks_legality() {
    let black = rune_runtime::shogi::legal::legal_moves(START).expect("gen");
    assert!(rune_runtime::shogi::legal::apply_move(START, "D99p").is_err());
    assert!(rune_runtime::shogi::legal::apply_move(START, "8080").is_err());
    let nxt = rune_runtime::shogi::legal::apply_move(START, &black[0]).expect("apply");
    let b = rune_runtime::shogi::ShogiBoard::parse_sfen(&nxt).expect("parse");
    assert_eq!(b.stm, rune_runtime::shogi::SWHITE);
    assert_eq!(b.move_no, 2);
}

#[test]
fn must_and_optional_promote() {
    let s = "9/1P7/9/9/9/9/9/9/4K4 b - 1";
    let moves = rune_runtime::shogi::legal::legal_moves(s).expect("gen");
    assert!(moves.contains(&"1001+".to_string()));
    assert!(!moves.contains(&"1001".to_string()));
    let s2 = "9/9/9/4P4/9/9/9/9/4K4 b - 1";
    let m2 = rune_runtime::shogi::legal::legal_moves(s2).expect("gen");
    assert!(m2.contains(&"3122".to_string()));
    assert!(m2.contains(&"3122+".to_string()));
}

#[test]
fn drop_restrictions() {
    let s = "9/9/9/9/4P4/9/9/9/4K4 b P 1";
    let moves = rune_runtime::shogi::legal::legal_moves(s).expect("gen");
    for m in &moves {
        if m.starts_with('D') && m.ends_with('p') {
            let sq: usize = m[1..m.len() - 1].parse().unwrap();
            assert_ne!(sq % 9, 4);
            assert_ne!(sq / 9, 0);
        }
    }
    let mate = "4k4/9/9/9/9/9/9/9/4K4 b P 1";
    let mm = rune_runtime::shogi::legal::legal_moves(mate).expect("gen");
    assert!(!mm.contains(&"D76p".to_string()));
}

#[test]
fn mate_is_loss() {
    let mate = "4k4/4G4/4G4/9/9/9/9/9/4K4 w - 1";
    let moves = rune_runtime::shogi::legal::legal_moves(mate).expect("gen");
    assert!(moves.is_empty());
    assert_eq!(rune_runtime::shogi::legal::game_result(mate).expect("res"), "1-0");
    assert_eq!(rune_runtime::shogi::legal::game_result(START).expect("res"), "*");
}

#[test]
fn decode_roundtrip() {
    let mv = rune_runtime::shogi::legal::decode_move("4847+").expect("dec");
    assert_eq!(rune_runtime::shogi::legal::encode_move(mv), "4847+");
    let d = rune_runtime::shogi::legal::decode_move("D56p").expect("dec");
    assert_eq!(rune_runtime::shogi::legal::encode_move(d), "D56p");
}
