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

use std::path::Path;

use shakmaty::{Chess, FromSetup, Position};

use crate::pipeline::{PipelineConfig, StageStats};
use crate::error::{Error, Result};
use crate::record::Record;

pub const HEADER_LEN: usize = 32;
pub const ENTRY_LEN: usize = 4;
pub const MAX_GAME_PLIES: usize = 2048;

#[derive(Debug, Clone, Default)]
pub struct MarlinStats {
    pub games: usize,
    pub positions: usize,
    pub skipped_games: usize,
}

fn cp_to_value_stm(cp_white: i32, stm_white: bool) -> f32 {
    let c = (cp_white as f32).clamp(-10000.0, 10000.0);
    let v = 2.0 / (1.0 + 10.0_f32.powf(-c / 400.0)) - 1.0;
    if stm_white {
        v
    } else {
        -v
    }
}

fn result_to_wdl_stm(result_white: u8, stm_white: bool) -> u8 {
    if stm_white {
        2 - result_white
    } else {
        result_white
    }
}

fn sq_name(sq: usize) -> String {
    let f = (sq % 8) as u8 + b'a';
    let r = (sq / 8) as u8 + b'1';
    String::from_utf8(vec![f, r]).unwrap_or_default()
}

fn piece_char(code: u8) -> Option<char> {
    match code {
        0 => Some('p'),
        1 => Some('n'),
        2 => Some('b'),
        3 => Some('r'),
        4 => Some('q'),
        5 => Some('k'),
        _ => None,
    }
}

struct DecodedBoard {
    fen: String,
}

fn decode_board(h: &[u8]) -> Result<DecodedBoard> {
    if h.len() < HEADER_LEN {
        return Err(Error::Truncated);
    }
    let occ = u64::from_le_bytes(h[0..8].try_into().map_err(|_| Error::Truncated)?);
    let pcs = &h[8..24];
    let stm_ep = h[24];
    let hmc = h[25];
    let fmn = u16::from_le_bytes([h[26], h[27]]);
    let squares: Vec<usize> = (0..64).filter(|s| occ >> s & 1 == 1).collect();
    if squares.is_empty() || squares.len() > 32 {
        return Err(Error::BadFormat("marlin bad occupancy".to_string()));
    }
    let stm_white = stm_ep >> 7 == 0;
    let mut grid: [Option<(char, bool)>; 64] = [None; 64];
    let mut wk = 0;
    let mut bk = 0;
    let mut kf: [Option<usize>; 2] = [None, None];
    let mut rook_files: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    for (i, sq) in squares.iter().enumerate() {
        let nib = (pcs[i / 2] >> ((i % 2) * 4)) & 0xF;
        let color = ((nib >> 3) & 1) as usize;
        let code = nib & 7;
        if code == 7 {
            return Err(Error::BadFormat("marlin bad piece code".to_string()));
        }
        if code == 5 {
            if color == 0 {
                wk += 1;
            } else {
                bk += 1;
            }
            kf[color] = Some(sq % 8);
        }
        if code == 6 {
            rook_files[color].push(sq % 8);
        }
        let base = if code == 6 { 'r' } else { piece_char(code).ok_or_else(|| Error::BadFormat("marlin bad piece".to_string()))? };
        let ch = if color == 0 { base.to_ascii_uppercase() } else { base };
        grid[*sq] = Some((ch, code == 6));
    }
    if wk != 1 || bk != 1 {
        return Err(Error::BadFormat("marlin bad king count".to_string()));
    }
    let mut castle = String::new();
    for color in [0usize, 1] {
        let Some(king_file) = kf[color] else {
            return Err(Error::BadFormat("marlin missing king".to_string()));
        };
        for rf in &rook_files[color] {
            if *rf != 0 && *rf != 7 {
                return Err(Error::BadFormat("marlin nonstandard castling".to_string()));
            }
            if king_file != 4 {
                return Err(Error::BadFormat("marlin nonstandard castling".to_string()));
            }
            let mark = if *rf > king_file {
                if color == 0 { 'K' } else { 'k' }
            } else if color == 0 {
                'Q'
            } else {
                'q'
            };
            castle.push(mark);
        }
    }
    let mut cs: String = ["K", "Q", "k", "q"].iter().filter(|m| castle.contains(**m)).map(|s| s.to_string()).collect();
    if cs.is_empty() {
        cs = "-".to_string();
    }
    let mut place = String::new();
    for rank in (0..8).rev() {
        let mut empty = 0;
        for file in 0..8 {
            match grid[rank * 8 + file] {
                Some((ch, _)) => {
                    if empty > 0 {
                        place.push_str(&empty.to_string());
                        empty = 0;
                    }
                    place.push(ch);
                }
                None => empty += 1,
            }
        }
        if empty > 0 {
            place.push_str(&empty.to_string());
        }
        if rank > 0 {
            place.push('/');
        }
    }
    let ep_raw = (stm_ep & 0x7F) as usize;
    let ep = if ep_raw >= 64 { "-".to_string() } else { sq_name(ep_raw) };
    let fen = format!("{} {} {} {} {} {}", place, if stm_white { "w" } else { "b" }, cs, ep, hmc, fmn.max(1));
    Ok(DecodedBoard { fen })
}

fn legal_match(pos: &Chess, raw: u16) -> Option<shakmaty::Move> {
    let from = (raw & 63) as usize;
    let to = ((raw >> 6) & 63) as usize;
    let promo = ((raw >> 12) & 3) as usize;
    let typ = raw >> 14;
    if from >= 64 || to >= 64 {
        return None;
    }
    let uci_text = if typ == 2 {
        let rank = from / 8;
        let dest_file = if to % 8 > from % 8 { 6 } else { 2 };
        format!("{}{}", sq_name(from), sq_name(rank * 8 + dest_file))
    } else if typ == 3 {
        let pc = ["n", "b", "r", "q"][promo];
        format!("{}{}{}", sq_name(from), sq_name(to), pc)
    } else {
        format!("{}{}", sq_name(from), sq_name(to))
    };
    let uci: shakmaty::uci::UciMove = uci_text.parse().ok()?;
    uci.to_move(pos).ok()
}

pub fn import_marlin(path: &Path, config: &PipelineConfig, stats: &mut StageStats) -> Result<Vec<Record>> {
    let data = std::fs::read(path)?;
    let mut out = Vec::new();
    let mut cursor = 0usize;
    let mut game_idx = 0usize;
    let mut rep = MarlinStats::default();
    let every = config.every_plies.max(1);
    while cursor < data.len() {
        if data.len() - cursor < HEADER_LEN {
            break;
        }
        let header = &data[cursor..cursor + HEADER_LEN];
        cursor += HEADER_LEN;
        let eval0 = i16::from_le_bytes([header[28], header[29]]);
        let result = header[30];
        if result > 2 {
            rep.skipped_games += 1;
            while cursor + ENTRY_LEN <= data.len() {
                let term = &data[cursor..cursor + ENTRY_LEN];
                cursor += ENTRY_LEN;
                if term == [0, 0, 0, 0] {
                    break;
                }
            }
            game_idx += 1;
            continue;
        }
        let mut entries: Vec<(u16, i16)> = Vec::new();
        let mut ok = false;
        for _ in 0..MAX_GAME_PLIES {
            if cursor + ENTRY_LEN > data.len() {
                break;
            }
            let e = &data[cursor..cursor + ENTRY_LEN];
            cursor += ENTRY_LEN;
            if e == [0, 0, 0, 0] {
                ok = true;
                break;
            }
            entries.push((u16::from_le_bytes([e[0], e[1]]), i16::from_le_bytes([e[2], e[3]])));
        }
        if !ok {
            rep.skipped_games += 1;
            game_idx += 1;
            continue;
        }
        let game_id = format!("marlin_{game_idx}");
        game_idx += 1;
        let fen0 = match decode_board(header) {
            Ok(b) => b.fen,
            Err(_) => {
                rep.skipped_games += 1;
                continue;
            }
        };
        let mut pos = match shakmaty::fen::Fen::from_ascii(fen0.as_bytes()) {
            Ok(setup) => match Chess::from_setup(setup.into_setup(), shakmaty::CastlingMode::Standard) {
                Ok(p) => p,
                Err(_) => {
                    rep.skipped_games += 1;
                    continue;
                }
            },
            Err(_) => {
                rep.skipped_games += 1;
                continue;
            }
        };
        let mut game_recs: Vec<Record> = Vec::new();
        let mut ply: u16 = 0;
        let mut failed = false;
        let push = |pos: &Chess, ply: u16, score_cp: i32, out: &mut Vec<Record>| -> bool {
            if !(ply as usize).is_multiple_of(every) {
                return true;
            }
            let fen = shakmaty::fen::Fen::from_position(pos, shakmaty::EnPassantMode::Legal).to_string();
            let stm_white = pos.turn() == shakmaty::Color::White;
            let rec = match Record::from_fen(&fen, &game_id, ply, config.source_id, config.strict_identity) {
                Ok(r) => r,
                Err(_) => return false,
            };
            out.push(rec.with_teacher(cp_to_value_stm(score_cp, stm_white), result_to_wdl_stm(result, stm_white), score_cp, true));
            true
        };
        if entries.is_empty() {
            let stm_white = pos.turn() == shakmaty::Color::White;
            if let Ok(r) = Record::from_fen(&fen0, &game_id, 0, config.source_id, config.strict_identity) {
                game_recs.push(r.with_teacher(cp_to_value_stm(eval0 as i32, stm_white), result_to_wdl_stm(result, stm_white), eval0 as i32, true));
            }
        } else {
            for (raw, score) in &entries {
                if ply as usize >= config.max_plies {
                    break;
                }
                if !push(&pos, ply, *score as i32, &mut game_recs) {
                    failed = true;
                    break;
                }
                match legal_match(&pos, *raw) {
                    Some(m) => pos.play_unchecked(m),
                    None => {
                        failed = true;
                        break;
                    }
                }
                ply = ply.saturating_add(1);
            }
        }
        if failed {
            rep.skipped_games += 1;
            continue;
        }
        rep.games += 1;
        rep.positions += game_recs.len();
        stats.input += game_recs.len();
        out.extend(game_recs);
    }
    stats.output = out.len();
    *stats.reasons.entry("marlin_skipped_games".to_string()).or_insert(0) += rep.skipped_games;
    Ok(out)
}
