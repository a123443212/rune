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

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::error::{Error, Result};
use crate::record::Record;

use super::{PipelineConfig, StageStats};

struct PgnVisitor {
    tags: Vec<(String, String)>,
    moves: Vec<pgn_reader::SanPlus>,
}

impl PgnVisitor {
    fn new() -> PgnVisitor {
        PgnVisitor {
            tags: Vec::new(),
            moves: Vec::new(),
        }
    }
}

impl pgn_reader::Visitor for PgnVisitor {
    type Tags = ();
    type Movetext = ();
    type Output = Option<(Option<String>, Vec<pgn_reader::SanPlus>)>;

    fn begin_tags(&mut self) -> std::ops::ControlFlow<Self::Output, Self::Tags> {
        self.tags.clear();
        self.moves.clear();
        std::ops::ControlFlow::Continue(())
    }

    fn tag(
        &mut self,
        _tags: &mut Self::Tags,
        name: &[u8],
        value: pgn_reader::RawTag<'_>,
    ) -> std::ops::ControlFlow<Self::Output> {
        self.tags.push((
            String::from_utf8_lossy(name).to_string(),
            String::from_utf8_lossy(value.0).to_string(),
        ));
        std::ops::ControlFlow::Continue(())
    }

    fn begin_movetext(
        &mut self,
        _tags: Self::Tags,
    ) -> std::ops::ControlFlow<Self::Output, Self::Movetext> {
        std::ops::ControlFlow::Continue(())
    }

    fn san(
        &mut self,
        _movetext: &mut Self::Movetext,
        san_plus: pgn_reader::SanPlus,
    ) -> std::ops::ControlFlow<Self::Output> {
        self.moves.push(san_plus);
        std::ops::ControlFlow::Continue(())
    }

    fn end_game(&mut self, _movetext: Self::Movetext) -> Self::Output {
        let fen = self
            .tags
            .iter()
            .find(|(key, _)| key == "FEN")
            .map(|(_, value)| value.trim_matches('"').to_string());
        Some((fen, std::mem::take(&mut self.moves)))
    }
}

pub fn ingest_pgn(
    path: &Path,
    config: &PipelineConfig,
    stats: &mut StageStats,
) -> Result<Vec<Record>> {
    ingest_pgn_threads(path, config, 1, stats)
}

pub fn ingest_pgn_threads(
    path: &Path,
    config: &PipelineConfig,
    threads: usize,
    stats: &mut StageStats,
) -> Result<Vec<Record>> {
    use shakmaty::{FromSetup, Position};

    let file = File::open(path)?;
    let mut reader = pgn_reader::Reader::new(BufReader::new(file));
    let mut games: Vec<(String, Option<String>, Vec<String>)> = Vec::new();
    let mut game_idx = 0usize;
    while let Some(game) = reader
        .read_game(&mut PgnVisitor::new())
        .map_err(|error| Error::BadPgn(error.to_string()))?
    {
        let (start_fen, moves) = match game {
            Some(game) => game,
            None => continue,
        };
        let san_moves: Vec<String> = moves.iter().map(|san| san.san.to_string()).collect();
        games.push((format!("pgn_{game_idx}"), start_fen, san_moves));
        game_idx += 1;
    }

    if games.is_empty() {
        stats.output = 0;
        return Ok(Vec::new());
    }

    let every = config.every_plies.max(1);
    let workers = threads.max(1).min(games.len());
    let chunk = games.len().div_ceil(workers);
    let mut output: Vec<Record> = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for part in games.chunks(chunk) {
            let config = config.clone();
            handles.push(scope.spawn(move || {
                let mut local: Vec<Record> = Vec::new();
                let mut local_input = 0usize;
                let mut local_rejected = 0usize;
                for (game_id, start_fen, san_moves) in part {
                    let mut position = match start_fen {
                        Some(fen) => match shakmaty::fen::Fen::from_ascii(fen.as_bytes()) {
                            Ok(setup) => match shakmaty::Chess::from_setup(
                                setup.into_setup(),
                                shakmaty::CastlingMode::Standard,
                            ) {
                                Ok(position) => position,
                                Err(_) => {
                                    local_rejected += 1;
                                    continue;
                                }
                            },
                            Err(_) => {
                                local_rejected += 1;
                                continue;
                            }
                        },
                        None => shakmaty::Chess::default(),
                    };
                    let mut ply: u16 = 0;
                    let mut push_position =
                        |position: &shakmaty::Chess, ply: u16, local: &mut Vec<Record>| {
                            if !(ply as usize).is_multiple_of(every) {
                                return;
                            }
                            let fen = shakmaty::fen::Fen::from_position(
                                position,
                                shakmaty::EnPassantMode::Legal,
                            )
                            .to_string();
                            local_input += 1;
                            match Record::from_fen(
                                &fen,
                                game_id,
                                ply,
                                config.source_id,
                                config.strict_identity,
                            ) {
                                Ok(record) => local.push(record),
                                Err(_) => local_rejected += 1,
                            }
                        };
                    push_position(&position, ply, &mut local);
                    for san_text in san_moves {
                        if ply as usize >= config.max_plies {
                            break;
                        }
                        let san: shakmaty::san::San = match san_text.parse() {
                            Ok(san) => san,
                            Err(_) => {
                                local_rejected += 1;
                                break;
                            }
                        };
                        let chess_move = match san.to_move(&position) {
                            Ok(chess_move) => chess_move,
                            Err(_) => {
                                local_rejected += 1;
                                break;
                            }
                        };
                        position.play_unchecked(chess_move);
                        ply = ply.saturating_add(1);
                        push_position(&position, ply, &mut local);
                    }
                }
                (local, local_input, local_rejected)
            }));
        }
        let mut results = Vec::new();
        for handle in handles {
            results.push(
                handle
                    .join()
                    .map_err(|_| Error::BadPgn("worker panic".to_string()))?,
            );
        }
        Ok::<_, Error>(results)
    })
    .map(|results| {
        for (local, local_input, local_rejected) in results {
            stats.input += local_input;
            stats.rejected += local_rejected;
            *stats.reasons.entry("record_build".to_string()).or_insert(0) += local_rejected;
            output.extend(local);
        }
        stats.output = output.len();
        output
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_pgn_returns_no_records() {
        let path = std::env::temp_dir().join(format!("rune-empty-pgn-{}", std::process::id()));
        std::fs::write(&path, []).unwrap();
        let mut stats = StageStats::default();
        let records = ingest_pgn(&path, &PipelineConfig::default(), &mut stats).unwrap();
        std::fs::remove_file(path).unwrap();
        assert!(records.is_empty());
        assert_eq!(stats.output, 0);
    }
}
