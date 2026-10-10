const START: &str = "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1";

#[test]
fn startpos_counts() {
    let red = rune_runtime::xiangqi::legal::legal_moves(START).expect("gen");
    assert_eq!(red.len(), 44);
    let mut sorted = red.clone();
    sorted.sort();
    assert_eq!(red, sorted);
    let nxt = rune_runtime::xiangqi::legal::apply_move(START, &red[0]).expect("apply");
    let black = rune_runtime::xiangqi::legal::legal_moves(&nxt).expect("gen");
    assert_eq!(black.len(), 44);
}

#[test]
fn apply_checks_legality() {
    let red = rune_runtime::xiangqi::legal::legal_moves(START).expect("gen");
    assert!(rune_runtime::xiangqi::legal::apply_move(START, "9999").is_err());
    let nxt = rune_runtime::xiangqi::legal::apply_move(START, &red[0]).expect("apply");
    assert_ne!(nxt, START);
    let b = rune_runtime::xiangqi::XiangqiBoard::parse_fen(&nxt).expect("parse");
    assert_eq!(b.stm, rune_runtime::xiangqi::XBLACK);
}

#[test]
fn mate_is_loss() {
    let mate = "3aka3/2H1P4/9/4R4/9/9/9/9/9/4K4 b - - 0 1";
    let moves = rune_runtime::xiangqi::legal::legal_moves(mate).expect("gen");
    assert!(moves.is_empty());
    assert_eq!(rune_runtime::xiangqi::legal::game_result(mate).expect("res"), "1-0");
    assert_eq!(rune_runtime::xiangqi::legal::game_result(START).expect("res"), "*");
}

#[test]
fn horse_leg_blocked() {
    let s = "3k5/9/9/9/3p5/3H5/9/9/9/5K3 w - - 0 1";
    let moves = rune_runtime::xiangqi::legal::legal_moves(s).expect("gen");
    assert!(!moves.contains(&"3958".to_string()));
    assert!(!moves.contains(&"3956".to_string()));
    assert!(moves.contains(&"3950".to_string()));
}

#[test]
fn flying_check_seen() {
    let fly = "4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1";
    let b = rune_runtime::xiangqi::XiangqiBoard::parse_fen(fly).expect("parse");
    assert!(rune_runtime::xiangqi::legal::king_in_check(&b, rune_runtime::xiangqi::XRED));
}
