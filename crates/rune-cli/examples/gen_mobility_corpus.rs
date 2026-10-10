// RUNE — Relational Unified Neural Evaluator
// Copyright (C) 2026 a123443212
//
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// This project is dual-licensed under the MIT License and the
// Apache License, Version 2.0. You may choose either license
// when using, copying, modifying, or distributing this software.
//
// MIT License: https://opensource.org/license/mit
// Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, this
// software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
// OR CONDITIONS OF ANY KIND, either express or implied.

use shakmaty::{Chess, EnPassantMode, Position};
use shakmaty::fen::Fen;
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.next() >> 33) as usize) % n.max(1)
    }
}
fn emit(fens: &mut Vec<String>, pos: &Chess) {
    fens.push(Fen::from_position(pos, EnPassantMode::Legal).to_string());
}
fn main() {
    let mut rng = Lcg(20261005);
    let mut fens: Vec<String> = Vec::new();
    for _ in 0..25 {
        let mut pos = Chess::default();
        emit(&mut fens, &pos);
        for ply in 1..=80 {
            let moves = pos.legal_moves();
            if moves.is_empty() {
                break;
            }
            let mv = moves[rng.below(moves.len())].clone();
            pos.play_unchecked(mv);
            if ply % 5 == 0 {
                emit(&mut fens, &pos);
            }
        }
    }
    let edges = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/P7/8/8/8/1k6/8/4K3 w - - 0 1",
        "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
        "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K2R b KQkq - 0 1",
        "7k/5Q2/6p1/6Pp/6P1/6K1/8/8 b - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b kq - 0 1",
        "2rr3k/pp3ppp/2n1pn2/q2p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w - - 0 12",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "N7/8/8/8/8/1k6/8/4K3 w - - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "k7/8/1K6/8/8/8/8/7R w - - 0 1",
    ];
    for e in edges {
        fens.push(e.to_string());
    }
    fens.sort();
    fens.dedup();
    for f in &fens {
        println!("{}", f);
    }
    eprintln!("corpus {}", fens.len());
}
