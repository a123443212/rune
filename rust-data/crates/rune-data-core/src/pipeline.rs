use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::board::Board;
use crate::error::{Error, Result};
use crate::format::{write_shard, Compression, ShardFile, ShardHeader};
use crate::hash;
use crate::record::Record;

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub seed: u64,
    pub strict_identity: bool,
    pub every_plies: usize,
    pub max_plies: usize,
    pub phase_cap_ratio: f64,
    pub source_id: u32,
    pub dataset_id: String,
    pub feature_version: String,
    pub compression: Compression,
    pub deterministic: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            seed: 42,
            strict_identity: false,
            every_plies: 4,
            max_plies: 120,
            phase_cap_ratio: 2.0,
            source_id: 0,
            dataset_id: "rune-clean-v07".to_string(),
            feature_version: "grouped_hkav2_fullthreats_v01".to_string(),
            compression: Compression::Raw,
            deterministic: true,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StageStats {
    pub input: usize,
    pub output: usize,
    pub rejected: usize,
    pub reasons: HashMap<String, usize>,
}

impl StageStats {
    pub fn reject(&mut self, reason: &str) {
        self.rejected += 1;
        *self.reasons.entry(reason.to_string()).or_insert(0) += 1;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineReport {
    pub dataset_id: String,
    pub seed: u64,
    pub stages: HashMap<String, StageStats>,
    pub phase_counts: [usize; 3],
    pub material_mean: f64,
    pub throughput_pos_per_sec: f64,
    pub elapsed_sec: f64,
    pub estimated_rss_mb: f64,
}

pub fn read_jsonl_pool(path: &Path) -> Result<Vec<PoolRecord>> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| Error::Io(e.to_string()))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).map_err(|e| Error::BadFormat(e.to_string()))?;
        out.push(PoolRecord::from_json(&v)?);
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct PoolRecord {
    pub fen: String,
    pub game_id: String,
    pub ply: u16,
    pub source_id: u32,
    pub teacher_value: Option<f32>,
    pub teacher_wdl: Option<u8>,
    pub teacher_cp: Option<i32>,
    pub perspective_stm: bool,
}

impl PoolRecord {
    pub fn from_json(v: &serde_json::Value) -> Result<PoolRecord> {
        let get = |k: &str| {
            v.get(k)
                .and_then(|x| x.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| Error::BadFormat(format!("pool record missing {k}")))
        };
        Ok(PoolRecord {
            fen: get("fen")?,
            game_id: v
                .get("game_id")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string(),
            ply: v.get("ply").and_then(|x| x.as_u64()).unwrap_or(8) as u16,
            source_id: v.get("source_id").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
            teacher_value: v
                .get("teacher_v")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32),
            teacher_wdl: v.get("teacher_w").and_then(|x| x.as_u64()).map(|x| x as u8),
            teacher_cp: v
                .get("teacher_cp")
                .and_then(|x| x.as_i64())
                .map(|x| x as i32),
            perspective_stm: v
                .get("value_perspective")
                .and_then(|x| x.as_str())
                .map(|s| s == "side_to_move")
                .unwrap_or(true),
        })
    }
}

pub fn pool_to_records(
    pool: &[PoolRecord],
    cfg: &PipelineConfig,
    stats: &mut StageStats,
) -> Vec<Record> {
    pool_to_records_with_ply(pool, cfg, 0, stats)
}

pub fn pool_to_records_with_ply(
    pool: &[PoolRecord],
    cfg: &PipelineConfig,
    min_ply: u32,
    stats: &mut StageStats,
) -> Vec<Record> {
    let mut out = Vec::new();
    for r in pool {
        stats.input += 1;
        if (r.ply as u32) < min_ply {
            stats.reject("min_ply");
            continue;
        }
        let board = match Board::parse_fen(&r.fen) {
            Ok(b) => b,
            Err(_) => {
                stats.reject("fen_parse");
                continue;
            }
        };
        if !valid_kings(&board) {
            stats.reject("king_count");
            continue;
        }
        if has_backrank_pawn(&board) {
            stats.reject("pawn_backrank");
            continue;
        }
        let gid = if r.game_id.is_empty() {
            &r.fen
        } else {
            &r.game_id
        };
        let mut rec = match Record::from_fen(&r.fen, gid, r.ply, r.source_id, cfg.strict_identity) {
            Ok(rec) => rec,
            Err(_) => {
                stats.reject("record_build");
                continue;
            }
        };
        if let (Some(v), Some(w)) = (r.teacher_value, r.teacher_wdl) {
            rec = rec.with_teacher(v, w, r.teacher_cp.unwrap_or(0), r.perspective_stm);
        }
        out.push(rec);
    }
    stats.output = out.len();
    out
}

fn valid_kings(b: &Board) -> bool {
    let mut wk = 0;
    let mut bk = 0;
    for c in b.sq.iter().flatten() {
        if c.kind == crate::board::KING {
            if c.color == crate::board::WHITE {
                wk += 1;
            } else {
                bk += 1;
            }
        }
    }
    wk == 1 && bk == 1
}

fn has_backrank_pawn(b: &Board) -> bool {
    for sq in 0..64 {
        if let Some(c) = b.sq[sq] {
            if c.kind == crate::board::PAWN {
                let r = crate::board::sq_rank(sq);
                if r == 0 || r == 7 {
                    return true;
                }
            }
        }
    }
    false
}

pub fn dedup_in_memory(records: Vec<Record>, stats: &mut StageStats) -> Vec<Record> {
    let mut seen: HashSet<u64> = HashSet::with_capacity(records.len());
    let mut out = Vec::with_capacity(records.len());
    for r in records {
        stats.input += 1;
        if seen.insert(r.identity) {
            out.push(r);
        } else {
            stats.reject("duplicate");
        }
    }
    stats.output = out.len();
    out
}

pub fn dedup_external(
    records: Vec<Record>,
    partitions: usize,
    tmp: &Path,
    stats: &mut StageStats,
) -> Result<Vec<Record>> {
    std::fs::create_dir_all(tmp)?;
    let mut writers: Vec<BufWriter<File>> = Vec::with_capacity(partitions);
    for p in 0..partitions {
        writers.push(BufWriter::new(File::create(
            tmp.join(format!("part-{p:04}.bin")),
        )?));
    }
    for r in &records {
        stats.input += 1;
        let p = (r.identity % partitions as u64) as usize;
        let mut buf = Vec::new();
        write_shard(
            &ShardHeader {
                feature_version: String::new(),
                compression: Compression::Raw,
                record_count: 1,
                shard_id: 0,
                shard_count: 1,
                dataset_id: String::new(),
            },
            std::slice::from_ref(r),
            &mut buf,
        )?;
        let len = buf.len() as u64;
        writers[p].write_all(&len.to_le_bytes())?;
        writers[p].write_all(&buf)?;
    }
    for w in &mut writers {
        w.flush()?;
    }
    drop(writers);
    let mut out = Vec::new();
    for p in 0..partitions {
        let path = tmp.join(format!("part-{p:04}.bin"));
        let data = std::fs::read(&path)?;
        let mut pos = 0usize;
        let mut seen: HashSet<u64> = HashSet::new();
        while pos + 8 <= data.len() {
            let mut lb = [0u8; 8];
            lb.copy_from_slice(&data[pos..pos + 8]);
            let len = u64::from_le_bytes(lb) as usize;
            pos += 8;
            if pos + len > data.len() {
                return Err(Error::BadFormat("partition truncated".to_string()));
            }
            let f = ShardFile::parse(&data[pos..pos + len])?;
            pos += len;
            let recs = f.read_all()?;
            for r in recs {
                if seen.insert(r.identity) {
                    out.push(r);
                } else {
                    stats.reject("duplicate");
                }
            }
        }
        let _ = std::fs::remove_file(&path);
    }
    out.sort_by_key(|a| (a.game_hash, a.ply, a.identity));
    stats.output = out.len();
    Ok(out)
}

#[derive(Debug, Clone, Default)]
pub struct FilterConfig {
    pub require_teacher: bool,
    pub min_abs_value: Option<f32>,
    pub phases: Option<Vec<u8>>,
    pub max_imbalance_bucket: Option<u8>,
    pub min_ply: u32,
}

pub fn filter_records(
    records: Vec<Record>,
    cfg: &FilterConfig,
    stats: &mut StageStats,
) -> Vec<Record> {
    let mut out = Vec::new();
    for r in records {
        stats.input += 1;
        if (r.ply as u32) < cfg.min_ply {
            stats.reject("min_ply");
            continue;
        }
        if cfg.require_teacher && !r.has_teacher {
            stats.reject("no_teacher");
            continue;
        }
        if let Some(m) = cfg.min_abs_value {
            if r.teacher_value.abs() < m {
                stats.reject("value_range");
                continue;
            }
        }
        if let Some(ph) = &cfg.phases {
            if !ph.contains(&r.phase) {
                stats.reject("phase");
                continue;
            }
        }
        if let Some(mb) = cfg.max_imbalance_bucket {
            if r.imbalance_bucket > mb {
                stats.reject("material");
                continue;
            }
        }
        out.push(r);
    }
    stats.output = out.len();
    out
}

pub fn phase_balance(records: Vec<Record>, max_ratio: f64, stats: &mut StageStats) -> Vec<Record> {
    let mut buckets: [Vec<Record>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for r in records {
        stats.input += 1;
        buckets[r.phase.min(2) as usize].push(r);
    }
    let counts = [buckets[0].len(), buckets[1].len(), buckets[2].len()];
    let nonzero: Vec<usize> = counts.iter().cloned().filter(|&c| c > 0).collect();
    if nonzero.is_empty() {
        stats.output = 0;
        return Vec::new();
    }
    let smallest = *nonzero.iter().min().unwrap();
    let target = *nonzero.iter().max().unwrap();
    let cap = if target as f64 > smallest as f64 * max_ratio {
        (smallest as f64 * max_ratio) as usize
    } else {
        target
    };
    let mut out = Vec::new();
    for b in buckets.iter_mut() {
        if b.len() > cap {
            stats.rejected += b.len() - cap;
            *stats.reasons.entry("balance_cap".to_string()).or_insert(0) += b.len() - cap;
        }
        out.extend(b.drain(..cap.min(b.len())));
    }
    stats.output = out.len();
    out
}

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }
    pub fn step(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.step() % (n as u64).max(1)) as usize
    }
}

pub fn sample_uniform(records: Vec<Record>, n: usize, seed: u64) -> Vec<Record> {
    let mut rng = Rng::new(seed);
    let mut idx: Vec<usize> = (0..records.len()).collect();
    for i in (1..idx.len()).rev() {
        let j = rng.below(i + 1);
        idx.swap(i, j);
    }
    let mut out = Vec::with_capacity(n.min(records.len()));
    let recs = records;
    for k in idx.into_iter().take(n.min(recs.len())) {
        out.push(recs[k].clone());
    }
    let _ = recs;
    out
}

pub fn sample_stratified(records: Vec<Record>, n: usize, seed: u64) -> Vec<Record> {
    let mut buckets: [Vec<Record>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for r in records {
        buckets[r.phase.min(2) as usize].push(r);
    }
    let per = n / 3;
    let mut out = Vec::new();
    let mut rng = Rng::new(seed);
    for b in buckets.iter_mut() {
        for i in (1..b.len()).rev() {
            let j = rng.below(i + 1);
            b.swap(i, j);
        }
        out.extend(b.drain(..per.min(b.len())));
    }
    out
}

pub fn game_split(
    records: Vec<Record>,
    train_frac: f64,
    val_frac: f64,
    seed: u64,
) -> (Vec<Record>, Vec<Record>, Vec<Record>) {
    let mut games: HashMap<u64, Vec<Record>> = HashMap::new();
    for r in records {
        games.entry(r.game_hash).or_default().push(r);
    }
    let mut gids: Vec<u64> = games.keys().cloned().collect();
    gids.sort_unstable();
    let mut train = Vec::new();
    let mut val = Vec::new();
    let mut test = Vec::new();
    for gid in gids {
        let h = hash::fnv1a(&[seed.to_le_bytes(), gid.to_le_bytes()].concat());
        let x = (h % 10000) as f64 / 10000.0;
        let bucket = if x < train_frac {
            &mut train
        } else if x < train_frac + val_frac {
            &mut val
        } else {
            &mut test
        };
        bucket.extend(games.remove(&gid).unwrap());
    }
    if val.is_empty()
        && train
            .iter()
            .map(|r| r.game_hash)
            .collect::<HashSet<_>>()
            .len()
            > 1
    {
        let mut seen_order: Vec<u64> = Vec::new();
        for r in &train {
            if !seen_order.contains(&r.game_hash) {
                seen_order.push(r.game_hash);
            }
        }
        if let Some(&last) = seen_order.last() {
            let mut rest = Vec::new();
            for r in train {
                if r.game_hash == last {
                    val.push(r);
                } else {
                    rest.push(r);
                }
            }
            train = rest;
        }
    }
    (train, val, test)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub dataset: String,
    pub schema: u32,
    pub feature_version: String,
    pub shards: usize,
    pub total_records: usize,
    pub hash: String,
    pub compression: String,
    pub seed: u64,
    pub shard_files: Vec<String>,
}

pub fn write_shards(
    records: &[Record],
    out_dir: &Path,
    dataset_id: &str,
    feature_version: &str,
    compression: Compression,
    per_shard: usize,
    seed: u64,
) -> Result<Manifest> {
    std::fs::create_dir_all(out_dir)?;
    let mut order: Vec<usize> = (0..records.len()).collect();
    order.sort_by_key(|&i| records[i].identity);
    let mut files = Vec::new();
    let mut shard_hashes: Vec<u64> = Vec::new();
    let n_shards = (records.len() + per_shard - 1) / per_shard.max(1);
    for (s, chunk) in order.chunks(per_shard.max(1)).enumerate() {
        let recs: Vec<Record> = chunk.iter().map(|&i| records[i].clone()).collect();
        let header = ShardHeader {
            feature_version: feature_version.to_string(),
            compression,
            record_count: recs.len() as u64,
            shard_id: s as u32,
            shard_count: n_shards as u32,
            dataset_id: dataset_id.to_string(),
        };
        let mut buf = Vec::new();
        write_shard(&header, &recs, &mut buf)?;
        let name = format!("shard-{s:05}");
        std::fs::write(out_dir.join(&name), &buf)?;
        shard_hashes.push(hash::fnv1a(&buf));
        files.push(name);
    }
    let mut acc = String::new();
    for h in &shard_hashes {
        acc.push_str(&format!("{h:016x}"));
    }
    acc.push_str(dataset_id);
    acc.push_str(&seed.to_string());
    Ok(Manifest {
        dataset: dataset_id.to_string(),
        schema: crate::format::SCHEMA_VERSION,
        feature_version: feature_version.to_string(),
        shards: n_shards,
        total_records: records.len(),
        hash: format!("{:016x}", hash::fnv1a(acc.as_bytes())),
        compression: format!("{compression:?}").to_lowercase(),
        seed,
        shard_files: files,
    })
}

pub fn read_shard_file(path: &Path) -> Result<Vec<Record>> {
    let data = std::fs::read(path)?;
    let f = ShardFile::parse(&data)?;
    f.read_all()
}

pub fn phase_counts(records: &[Record]) -> [usize; 3] {
    let mut c = [0usize; 3];
    for r in records {
        c[r.phase.min(2) as usize] += 1;
    }
    c
}

pub fn near_duplicate_stats(records: &[Record]) -> (usize, usize, usize) {
    let mut by_game: HashMap<u64, Vec<u64>> = HashMap::new();
    for r in records {
        by_game.entry(r.game_hash).or_default().push(r.identity);
    }
    let mut local_dup = 0usize;
    let mut clusters = 0usize;
    for ids in by_game.values() {
        let mut seen = HashSet::new();
        let mut dup = false;
        for id in ids {
            if !seen.insert(*id) {
                local_dup += 1;
                dup = true;
            }
        }
        if dup {
            clusters += 1;
        }
    }
    let mut counts: HashMap<u64, usize> = HashMap::new();
    for r in records {
        *counts.entry(r.identity).or_default() += 1;
    }
    let exact_dup: usize = counts.values().map(|&c| c.saturating_sub(1)).sum();
    (exact_dup, local_dup, clusters)
}

pub fn join_labels(
    records: Vec<Record>,
    labels: &[(u64, f32, u8, i32)],
    stats: &mut StageStats,
) -> (Vec<Record>, usize, usize) {
    let mut map: HashMap<u64, Vec<(f32, u8, i32)>> = HashMap::new();
    for (id, v, w, cp) in labels {
        map.entry(*id).or_default().push((*v, *w, *cp));
    }
    let mut out = Vec::new();
    let mut missing = 0usize;
    let mut dup_labels = 0usize;
    for mut r in records {
        stats.input += 1;
        match map.get(&r.identity) {
            None => {
                missing += 1;
                stats.reject("missing_label");
            }
            Some(v) if v.len() > 1 => {
                dup_labels += 1;
                stats.reject("duplicate_label");
            }
            Some(v) => {
                r = r.with_teacher(v[0].0, v[0].1, v[0].2, true);
                out.push(r);
            }
        }
    }
    stats.output = out.len();
    (out, missing, dup_labels)
}

pub fn write_report(path: &Path, report: &PipelineReport) -> Result<()> {
    let json = serde_json::to_string_pretty(report).map_err(|e| Error::BadFormat(e.to_string()))?;
    std::fs::write(path.with_extension("json"), &json)?;
    let mut md = String::new();
    md.push_str(&format!("# Pipeline report: {}\n\n", report.dataset_id));
    md.push_str(&format!("- seed: {}\n", report.seed));
    md.push_str(&format!(
        "- elapsed: {:.2}s, throughput: {:.0} pos/s\n",
        report.elapsed_sec, report.throughput_pos_per_sec
    ));
    md.push_str(&format!(
        "- estimated RSS: {:.1} MB\n",
        report.estimated_rss_mb
    ));
    md.push_str(&format!("- phases: {:?}\n", report.phase_counts));
    md.push_str(&format!("- mean material: {:.1}\n", report.material_mean));
    for (stage, s) in &report.stages {
        md.push_str(&format!(
            "\n## {stage}: in={} out={} rejected={} reasons={:?}\n",
            s.input, s.output, s.rejected, s.reasons
        ));
    }
    std::fs::write(path.with_extension("md"), md)?;
    Ok(())
}

pub fn rss_mb_estimate() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            if let Some(kb) = rest
                .split_whitespace()
                .next()
                .and_then(|x| x.parse::<f64>().ok())
            {
                return kb / 1024.0;
            }
        }
    }
    0.0
}

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
            .find(|(k, _)| k == "FEN")
            .map(|(_, v)| v.trim_matches('"').to_string());
        Some((fen, std::mem::take(&mut self.moves)))
    }
}

pub fn ingest_pgn(
    path: &Path,
    cfg: &PipelineConfig,
    stats: &mut StageStats,
) -> Result<Vec<Record>> {
    ingest_pgn_threads(path, cfg, 1, stats)
}

pub fn ingest_pgn_threads(
    path: &Path,
    cfg: &PipelineConfig,
    threads: usize,
    stats: &mut StageStats,
) -> Result<Vec<Record>> {
    use shakmaty::{FromSetup, Position};

    let f = File::open(path)?;
    let mut reader = pgn_reader::Reader::new(BufReader::new(f));
    let mut games: Vec<(String, Option<String>, Vec<String>)> = Vec::new();
    let mut game_idx = 0usize;
    while let Some(game) = reader
        .read_game(&mut PgnVisitor::new())
        .map_err(|e| Error::BadPgn(e.to_string()))?
    {
        let (start_fen, moves) = match game {
            Some(g) => g,
            None => continue,
        };
        let sans: Vec<String> = moves.iter().map(|s| s.san.to_string()).collect();
        games.push((format!("pgn_{game_idx}"), start_fen, sans));
        game_idx += 1;
    }

    let every = cfg.every_plies.max(1);
    let workers = threads.max(1).min(games.len().max(1));
    let chunk = games.len().div_ceil(workers);
    let mut out: Vec<Record> = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for part in games.chunks(chunk) {
            let cfg = cfg.clone();
            handles.push(scope.spawn(move || {
                let mut local: Vec<Record> = Vec::new();
                let mut local_in = 0usize;
                let mut local_rej = 0usize;
                for (gid, start_fen, sans) in part {
                    let mut pos = match start_fen {
                        Some(f) => match shakmaty::fen::Fen::from_ascii(f.as_bytes()) {
                            Ok(setup) => match shakmaty::Chess::from_setup(
                                setup.into_setup(),
                                shakmaty::CastlingMode::Standard,
                            ) {
                                Ok(p) => p,
                                Err(_) => {
                                    local_rej += 1;
                                    continue;
                                }
                            },
                            Err(_) => {
                                local_rej += 1;
                                continue;
                            }
                        },
                        None => shakmaty::Chess::default(),
                    };
                    let mut ply: u16 = 0;
                    let mut push_pos =
                        |pos: &shakmaty::Chess, ply: u16, local: &mut Vec<Record>| {
                            if !(ply as usize).is_multiple_of(every) {
                                return;
                            }
                            let fen = shakmaty::fen::Fen::from_position(
                                pos,
                                shakmaty::EnPassantMode::Legal,
                            )
                            .to_string();
                            local_in += 1;
                            match Record::from_fen(
                                &fen,
                                gid,
                                ply,
                                cfg.source_id,
                                cfg.strict_identity,
                            ) {
                                Ok(r) => local.push(r),
                                Err(_) => local_rej += 1,
                            }
                        };
                    push_pos(&pos, ply, &mut local);
                    for san_str in sans {
                        if ply as usize >= cfg.max_plies {
                            break;
                        }
                        let san: shakmaty::san::San = match san_str.parse() {
                            Ok(s) => s,
                            Err(_) => {
                                local_rej += 1;
                                break;
                            }
                        };
                        let mv = match san.to_move(&pos) {
                            Ok(m) => m,
                            Err(_) => {
                                local_rej += 1;
                                break;
                            }
                        };
                        pos.play_unchecked(mv);
                        ply = ply.saturating_add(1);
                        push_pos(&pos, ply, &mut local);
                    }
                }
                (local, local_in, local_rej)
            }));
        }
        let mut results = Vec::new();
        for h in handles {
            results.push(
                h.join()
                    .map_err(|_| Error::BadPgn("worker panic".to_string()))?,
            );
        }
        Ok::<_, Error>(results)
    })
    .map(|results| {
        for (local, local_in, local_rej) in results {
            stats.input += local_in;
            stats.rejected += local_rej;
            *stats.reasons.entry("record_build".to_string()).or_insert(0) += local_rej;
            out.extend(local);
        }
        stats.output = out.len();
        out
    })
}
