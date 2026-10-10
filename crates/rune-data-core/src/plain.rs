use std::path::Path;

use crate::pipeline::{PipelineConfig, StageStats};
use crate::error::Result;
use crate::record::Record;

#[derive(Debug, Clone, Default)]
pub struct PlainStats {
    pub lines: usize,
    pub positions: usize,
    pub skipped: usize,
}

pub struct PlainPosition {
    pub fen: String,
    pub score_cp: i32,
    pub result_white: Option<u8>,
}

fn cp_to_value_stm(cp: i32, stm_white: bool) -> f32 {
    let c = (cp as f32).clamp(-10000.0, 10000.0);
    let v = 2.0 / (1.0 + 10.0_f32.powf(-c / 400.0)) - 1.0;
    if stm_white {
        v
    } else {
        -v
    }
}

fn parse_result_token(tok: &str) -> Option<u8> {
    let t = tok.trim();
    if t.contains('.') {
        return match t {
            "1.0" => Some(2),
            "0.5" => Some(1),
            "0.0" => Some(0),
            _ => None,
        };
    }
    match t {
        "0" => Some(0),
        "1" => Some(1),
        "2" => Some(2),
        _ => None,
    }
}

fn parse_score_token(tok: &str) -> Option<i32> {
    if let Ok(v) = tok.trim().parse::<i32>() {
        return Some(v);
    }
    tok.trim().parse::<f32>().ok().map(|v| v.round() as i32)
}

fn parse_line(line: &str) -> Option<PlainPosition> {
    let t = line.trim();
    if t.is_empty() || t.starts_with('#') {
        return None;
    }
    if let Some(bar) = t.find('|') {
        let fen = t[..bar].trim().to_string();
        let rest: Vec<&str> = t[bar + 1..].split('|').collect();
        let score = parse_score_token(rest.first()?.trim())?;
        let result_white = if rest.len() > 1 { Some(parse_result_token(rest[1])?) } else { None };
        return Some(PlainPosition { fen, score_cp: score, result_white });
    }
    let toks: Vec<&str> = t.split_whitespace().collect();
    if toks.len() < 7 {
        return None;
    }
    let fen = toks[..6].join(" ");
    let tail = &toks[6..];
    if tail.len() == 1 {
        Some(PlainPosition { fen, score_cp: parse_score_token(tail[0])?, result_white: None })
    } else if tail.len() == 2 {
        Some(PlainPosition { fen, score_cp: parse_score_token(tail[0])?, result_white: Some(parse_result_token(tail[1])?) })
    } else {
        None
    }
}

pub fn parse_plain_file(path: &Path) -> Result<(Vec<PlainPosition>, PlainStats)> {
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    let mut st = PlainStats::default();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        st.lines += 1;
        match parse_line(t) {
            Some(p) => {
                st.positions += 1;
                out.push(p);
            }
            None => st.skipped += 1,
        }
    }
    Ok((out, st))
}

fn wdl_from_value(v: f32) -> u8 {
    if v.abs() < 0.15 {
        1
    } else if v > 0.0 {
        0
    } else {
        2
    }
}

pub fn plain_to_records(positions: Vec<PlainPosition>, game_id: &str, config: &PipelineConfig, stats: &mut StageStats) -> Vec<Record> {
    let mut out = Vec::new();
    for (i, p) in positions.iter().enumerate() {
        let rec = match Record::from_fen(&p.fen, game_id, (i % 65536) as u16, config.source_id, config.strict_identity) {
            Ok(r) => r,
            Err(_) => {
                stats.rejected += 1;
                continue;
            }
        };
        let stm_white = rec.stm == crate::board::WHITE;
        let v = cp_to_value_stm(p.score_cp, stm_white);
        let w = match p.result_white {
            Some(r) => {
                if stm_white {
                    2 - r
                } else {
                    r
                }
            }
            None => wdl_from_value(v),
        };
        stats.input += 1;
        out.push(rec.with_teacher(v, w, p.score_cp, true));
    }
    stats.output = out.len();
    out
}

pub fn import_plain(path: &Path, config: &PipelineConfig, stats: &mut StageStats) -> Result<Vec<Record>> {
    let (positions, st) = parse_plain_file(path)?;
    *stats.reasons.entry("plain_skipped_lines".to_string()).or_insert(0) += st.skipped;
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("plain").to_string();
    Ok(plain_to_records(positions, &stem, config, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(name: &str, body: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("rune-plain-{name}-{}", std::process::id()));
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        p
    }

    #[test]
    fn parses_pipe_and_plain_lines() {
        let p = write_tmp("mix", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 25 | 1.0\nrnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1 30\n# comment\n\nbad line\n");
        let (v, st) = parse_plain_file(&p).unwrap();
        std::fs::remove_file(&p).unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(st.skipped, 1);
        assert_eq!(v[0].score_cp, 25);
        assert_eq!(v[0].result_white, Some(2));
        assert_eq!(v[1].result_white, None);
    }

    #[test]
    fn records_carry_stm_teachers() {
        let p = write_tmp("rec", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 100 | 1.0\n");
        let mut stats = StageStats::default();
        let recs = import_plain(&p, &PipelineConfig::default(), &mut stats).unwrap();
        std::fs::remove_file(&p).unwrap();
        assert_eq!(recs.len(), 1);
        assert!(recs[0].has_teacher);
        assert!(recs[0].teacher_value > 0.0);
        assert_eq!(recs[0].teacher_wdl, 0);
        assert_eq!(recs[0].teacher_cp, 100);
    }
}
