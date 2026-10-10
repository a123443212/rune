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

use super::board::{group_and_liberties, neighbors_of};

pub fn opponent_stone(stone: i8) -> i8 {
    if stone == 1 {
        return -1;
    }
    1
}

pub fn play_stone(board: &[i8], n: usize, sq: usize, stone: i8, ko: Option<usize>) -> Option<(Vec<i8>, Option<usize>, Vec<usize>)> {
    if board[sq] != 0 {
        return None;
    }
    if let Some(k) = ko {
        if sq == k {
            return None;
        }
    }
    let mut nb = board.to_vec();
    nb[sq] = stone;
    let opp = opponent_stone(stone);
    let mut captured: Vec<usize> = Vec::new();
    let mut seen = vec![false; n * n];
    for adj in neighbors_of(sq, n) {
        if nb[adj] != opp || seen[adj] {
            continue;
        }
        let (stones, libs) = group_and_liberties(&nb, adj, n);
        for s in &stones {
            seen[*s] = true;
        }
        if libs.is_empty() {
            captured.extend(stones);
        }
    }
    for s in &captured {
        nb[*s] = 0;
    }
    let (_, libs) = group_and_liberties(&nb, sq, n);
    if libs.is_empty() {
        return None;
    }
    let mut new_ko: Option<usize> = None;
    if captured.len() == 1 {
        let (stones, libs2) = group_and_liberties(&nb, sq, n);
        if stones.len() == 1 && libs2.len() == 1 {
            new_ko = Some(captured[0]);
        }
    }
    Some((nb, new_ko, captured))
}

pub fn legal_moves(board: &[i8], n: usize, stm: u8, ko: Option<usize>) -> Vec<i32> {
    let stone: i8 = if stm == 0 { 1 } else { -1 };
    let mut out = Vec::new();
    for sq in 0..n * n {
        if board[sq] != 0 {
            continue;
        }
        if let Some(k) = ko {
            if sq == k {
                continue;
            }
        }
        if play_stone(board, n, sq, stone, ko).is_some() {
            out.push(sq as i32);
        }
    }
    out.push(-1);
    out
}

pub fn apply_move(board: &[i8], n: usize, stm: u8, mv: i32, ko: Option<usize>) -> super::super::error::Result<(Vec<i8>, Option<usize>, Vec<usize>)> {
    if mv == -1 {
        return Ok((board.to_vec(), None, Vec::new()));
    }
    let stone: i8 = if stm == 0 { 1 } else { -1 };
    match play_stone(board, n, mv as usize, stone, ko) {
        Some(v) => Ok(v),
        None => Err(crate::error::RuntimeError::InvalidState("illegal go move".to_string())),
    }
}

pub fn territory_owner(board: &[i8], n: usize) -> Vec<i8> {
    let mut owner = vec![0i8; n * n];
    let mut done = vec![false; n * n];
    for sq in 0..n * n {
        if board[sq] != 0 || done[sq] {
            continue;
        }
        let mut region = Vec::new();
        let mut border: Vec<i8> = Vec::new();
        let mut stack = vec![sq];
        done[sq] = true;
        while let Some(cur) = stack.pop() {
            region.push(cur);
            let r = cur / n;
            let c = cur % n;
            let deltas: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
            for (dr, dc) in deltas {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr < 0 || nc < 0 || nr >= n as i32 || nc >= n as i32 {
                    continue;
                }
                let nsq = nr as usize * n + nc as usize;
                let v = board[nsq];
                if v == 0 && !done[nsq] {
                    done[nsq] = true;
                    stack.push(nsq);
                } else if v != 0 && !border.contains(&v) {
                    border.push(v);
                }
            }
        }
        let fill: i8 = if border.len() == 1 { border[0] } else { 0 };
        for s in region {
            owner[s] = fill;
        }
    }
    owner
}

pub fn area_score(board: &[i8], n: usize, komi: f32) -> f32 {
    let mut black = 0i32;
    let mut white = 0i32;
    for v in board.iter() {
        if *v == 1 {
            black += 1;
        } else if *v == -1 {
            white += 1;
        }
    }
    let owner = territory_owner(board, n);
    for sq in 0..n * n {
        if board[sq] != 0 {
            continue;
        }
        if owner[sq] == 1 {
            black += 1;
        } else if owner[sq] == -1 {
            white += 1;
        }
    }
    black as f32 - white as f32 - komi
}
