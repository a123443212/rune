use rune_runtime::go::{GoBoard, GoState, compute_context, extract_planes, extract_planes_v02, token_features_v02};

fn empty9() -> String {
    vec!["........."; 9].join("/") + " b - 7.5 1 0"
}

#[test]
fn v01_still_works() {
    let s = "........./........./........./........./........./........./........./........./......... b";
    let b = GoBoard::parse(s).expect("parse v01");
    assert_eq!(b.size, 9);
    let p = extract_planes(&b);
    assert_eq!(p.len(), 81);
    let c = compute_context(&b);
    assert_eq!(c.len(), 12);
}

#[test]
fn v02_empty_planes() {
    let st = GoState::parse(&empty9()).expect("parse");
    assert_eq!(st.board.size, 9);
    assert_eq!(st.ko, None);
    let p = extract_planes_v02(&st);
    assert_eq!(p.len(), 8 * 81);
    let mut own = 0.0f32;
    let mut opp = 0.0f32;
    let mut empty = 0.0f32;
    for sq in 0..81 {
        own += p[sq];
        opp += p[81 + sq];
        empty += p[2 * 81 + sq];
    }
    assert_eq!(own, 0.0);
    assert_eq!(opp, 0.0);
    assert_eq!(empty, 81.0);
    let ctx = st.context_v02();
    assert_eq!(ctx.len(), 12);
    assert_eq!(ctx[0], 0.0);
    assert_eq!(ctx[6], 0.0);
}

#[test]
fn v02_single_stone_liberties() {
    let mut rows = vec!["........."; 9];
    rows[4] = "....X....";
    let s = rows.join("/") + " b - 7.5 10 0";
    let st = GoState::parse(&s).expect("parse");
    let p = extract_planes_v02(&st);
    let sq = 4 * 9 + 4;
    assert_eq!(p[sq], 1.0);
    assert_eq!(p[5 * 81 + sq], 1.0);
    assert_eq!(p[3 * 81 + sq], 0.0);
    assert_eq!(p[4 * 81 + sq], 0.0);
    let feats = token_features_v02(&st);
    assert!(!feats.is_empty());
    for (g, i) in &feats {
        assert!(*g < 9);
        assert!((*i as usize) < rune_runtime::go::GO_VOCABS[*g as usize]);
    }
    assert_eq!(st.phase_v02(), 0);
}

#[test]
fn v02_suicide_excluded() {
    let n = 9usize;
    let mut board = vec![0i8; n * n];
    board[1] = -1;
    board[9] = -1;
    let legal = rune_runtime::go::legal_moves(&board, n, 0, None);
    assert!(!legal.contains(&0));
    assert!(legal.contains(&-1));
}

#[test]
fn v02_capture_works() {
    let n = 9usize;
    let mut board2 = vec![0i8; n * n];
    board2[1] = 1;
    board2[9] = 1;
    let res2 = rune_runtime::go::play_stone(&board2, n, 0, -1, None);
    assert!(res2.is_none());
    let mut board3 = vec![0i8; n * n];
    board3[0] = -1;
    board3[1] = 1;
    board3[9] = 1;
    board3[10] = 1;
    let res3 = rune_runtime::go::play_stone(&board3, n, 11, 1, None);
    assert!(res3.is_some());
}

#[test]
fn v02_score_empty_is_minus_komi() {
    let st = GoState::parse(&empty9()).expect("parse");
    let s = rune_runtime::go::area_score(&st.board.stones, 9, 7.5);
    assert_eq!(s, -7.5);
}

#[test]
fn v02_ko_roundtrip() {
    let mut rows = vec!["........."; 9];
    rows[4] = "....X....";
    let s = rows.join("/") + " w 5 7.5 20 1";
    let st = GoState::parse(&s).expect("parse");
    assert_eq!(st.ko, Some(5));
    assert_eq!(st.move_no, 20);
    assert_eq!(st.pass_no, 1);
    let enc = st.encode();
    let st2 = GoState::parse(&enc).expect("reparse");
    assert_eq!(st, st2);
    let ctx = st.context_v02();
    assert_eq!(ctx[6], 1.0);
}

#[test]
fn v02_legal_contains_pass() {
    let st = GoState::parse(&empty9()).expect("parse");
    let legal = rune_runtime::go::legal_moves(&st.board.stones, 9, 0, None);
    assert!(legal.contains(&-1));
    assert_eq!(legal.len(), 82);
}

#[test]
fn v02_phase_late() {
    let s = vec!["........."; 9].join("/") + " b - 7.5 70 0";
    let st = GoState::parse(&s).expect("parse");
    assert_eq!(st.phase_v02(), 2);
}
