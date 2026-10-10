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
}

impl Session {
    pub fn new(model: &str, depth: usize) -> Result<Session, String> {
        let ev = Evaluator::load(&PathBuf::from(model)).map_err(|e| e.to_string())?;
        Ok(Session { ev, fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(), depth, model_path: model.to_string() })
    }

    fn apply_position(&mut self, args: &[String]) {
        let mut pos = if !args.is_empty() && args[0] == "startpos" {
            Chess::default()
        } else if args.len() >= 7 && args[0] == "fen" {
            let fen_text = args[1..7].join(" ");
            match shakmaty::fen::Fen::from_ascii(fen_text.as_bytes()) {
                Ok(setup) => match Chess::from_setup(setup.into_setup(), shakmaty::CastlingMode::Standard) {
                    Ok(p) => p,
                    Err(_) => return,
                },
                Err(_) => return,
            }
        } else {
            return;
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
            return;
        }
        for m in moves {
            let uci: Result<shakmaty::uci::UciMove, _> = m.parse();
            let mv = match uci {
                Ok(u) => u.to_move(&pos),
                Err(_) => return,
            };
            match mv {
                Ok(mv) => pos.play_unchecked(mv),
                Err(_) => return,
            }
        }
        self.fen = shakmaty::fen::Fen::from_position(&pos, shakmaty::EnPassantMode::Legal).to_string();
    }

    fn run_go(&mut self, args: &[String]) {
        let mut depth = self.depth;
        let mut movetime: Option<u64> = None;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "depth" => {
                    if i + 1 < args.len() {
                        if let Ok(d) = args[i + 1].parse::<usize>() {
                            depth = d.clamp(1, 64);
                        }
                        i += 1;
                    }
                }
                "movetime" => {
                    if i + 1 < args.len() {
                        if let Ok(ms) = args[i + 1].parse::<u64>() {
                            movetime = Some(ms);
                        }
                        i += 1;
                    }
                }
                "wtime" | "btime" => {
                    if movetime.is_none() && i + 1 < args.len() {
                        if let Ok(ms) = args[i + 1].parse::<u64>() {
                            movetime = Some((ms / 20).clamp(10, 5000));
                        }
                        i += 1;
                    } else if i + 1 < args.len() {
                        i += 1;
                    }
                }
                "winc" | "binc" => {
                    i += 1;
                }
                _ => {}
            }
            i += 1;
        }
        let out = match movetime {
            Some(ms) => searcher::search_timed(&mut self.ev, &self.fen, ms, depth),
            None => searcher::search_depth(&mut self.ev, &self.fen, depth),
        };
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
                "position" => self.apply_position(&parts),
                "go" => self.run_go(&parts),
                "stop" => println!("bestmove 0000"),
                "quit" => break,
                _ => {}
            }
        }
    }
}
