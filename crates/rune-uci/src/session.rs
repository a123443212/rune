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

use std::io::BufRead;
use std::path::PathBuf;

use shakmaty::{Chess, FromSetup, Position};
use rune_runtime::evaluator::Evaluator;

use crate::searcher;

pub struct Session {
    pub ev: Evaluator,
    pub fen: String,
    pub depth: usize,
    pub model_path: String,
    last_best: String,
}

impl Session {
    pub fn new(model: &str, depth: usize) -> Result<Session, String> {
        let ev = Evaluator::load(&PathBuf::from(model)).map_err(|e| e.to_string())?;
        Ok(Session { ev, fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(), depth, model_path: model.to_string(), last_best: "0000".to_string() })
    }

    fn apply_position(&mut self, args: &[String]) -> Result<(), String> {
        let mut pos = if !args.is_empty() && args[0] == "startpos" {
            Chess::default()
        } else if args.len() >= 7 && args[0] == "fen" {
            let fen_text = args[1..7].join(" ");
            match shakmaty::fen::Fen::from_ascii(fen_text.as_bytes()) {
                Ok(setup) => match Chess::from_setup(setup.into_setup(), shakmaty::CastlingMode::Standard) {
                    Ok(p) => p,
                    Err(_) => return Err("bad fen position".to_string()),
                },
                Err(_) => return Err("bad fen".to_string()),
            }
        } else {
            return Err("bad position args".to_string());
        };
        let rest = if !args.is_empty() && args[0] == "startpos" {
            &args[1..]
        } else if args.len() >= 7 && args[0] == "fen" {
            &args[7..]
        } else {
            &[][..]
        };
        let mut moves: &[String] = &[];
        if rest.len() >= 2 && rest[0] == "moves" {
            moves = &rest[1..];
        } else if !rest.is_empty() {
            return Err("bad position suffix".to_string());
        }
        for m in moves {
            let uci: Result<shakmaty::uci::UciMove, _> = m.parse();
            let mv = match uci {
                Ok(u) => u.to_move(&pos),
                Err(_) => return Err(format!("bad uci {}", m)),
            };
            match mv {
                Ok(mv) => pos.play_unchecked(mv),
                Err(_) => return Err(format!("illegal {}", m)),
            }
        }
        self.fen = shakmaty::fen::Fen::from_position(&pos, shakmaty::EnPassantMode::Legal).to_string();
        Ok(())
    }

    fn run_go(&mut self, args: &[String]) {
        let mut depth = self.depth;
        let mut movetime: Option<u64> = None;
        let mut wtime: Option<u64> = None;
        let mut btime: Option<u64> = None;
        let mut winc: u64 = 0;
        let mut binc: u64 = 0;
        let mut i = 0;
        while i < args.len() {
            let key = args[i].as_str();
            let value = args.get(i + 1);
            let num = value.and_then(|v| v.parse::<u64>().ok());
            let consumes_value = value.is_some()
                && matches!(key, "depth" | "movetime" | "wtime" | "btime" | "winc" | "binc");
            match key {
                "depth" => {
                    if let Some(d) = num {
                        depth = (d as usize).clamp(1, 64);
                    }
                }
                "movetime" => movetime = num.or(movetime),
                "wtime" => wtime = num.or(wtime),
                "btime" => btime = num.or(btime),
                "winc" => winc = num.unwrap_or(winc),
                "binc" => binc = num.unwrap_or(binc),
                _ => {}
            }
            i += if consumes_value { 2 } else { 1 };
        }
        if movetime.is_none() {
            let stm_white = self.fen.split_whitespace().nth(1).map(|s| s == "w").unwrap_or(true);
            let (tc, inc) = if stm_white { (wtime, winc) } else { (btime, binc) };
            if let Some(ms) = tc {
                movetime = Some((ms / 20 + inc / 2).clamp(10, 5000));
            }
        }
        let out = match movetime {
            Some(ms) => searcher::search_timed(&mut self.ev, &self.fen, ms, depth),
            None => searcher::search_depth(&mut self.ev, &self.fen, depth),
        };
        self.last_best = out.bestmove.clone();
        let cp = searcher::value_to_cp(out.value);
        println!("info depth {} score cp {} nodes {} time {} pv {}", depth, cp, out.nodes, out.millis, out.bestmove);
        println!("bestmove {}", out.bestmove);
    }

    pub fn serve(&mut self) {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            let mut parts: Vec<String> = t.split_whitespace().map(|s| s.to_string()).collect();
            let cmd = parts.remove(0);
            match cmd.as_str() {
                "uci" => {
                    println!("id name RUNE {}", self.ev.arch_id());
                    println!("id author rune");
                    println!("option name Model type string default {}", self.model_path);
                    println!("uciok");
                }
                "isready" => println!("readyok"),
                "ucinewgame" => {
                    self.fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string();
                }
                "setoption" => {
                    if let Some(idx) = parts.iter().position(|s| s == "value") {
                        if parts.get(idx.saturating_sub(2)).map(|s| s.as_str()) == Some("Model") {
                            let p = parts[idx + 1..].join(" ");
                            if let Ok(ev) = Evaluator::load(&PathBuf::from(&p)) {
                                self.ev = ev;
                                self.model_path = p;
                            }
                        }
                    }
                }
                "position" => {
                    if let Err(e) = self.apply_position(&parts) {
                        println!("info string position error {}", e);
                    }
                }
                "go" => self.run_go(&parts),
                "stop" => println!("bestmove {}", self.last_best),
                "quit" => break,
                _ => {}
            }
        }
    }
}
